"""
Django signals for model_hub app.

Handles cross-app synchronization between prompt playground (model_hub)
and agent playground when PromptVersion changes occur.

Also handles cascade soft-delete when Dataset, PromptTemplate, or
PromptVersion are soft-deleted.
"""

import threading
import time
from threading import local

import structlog
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver

logger = structlog.get_logger(__name__)

# Thread-local storage for capturing old state before save
_thread_locals = local()

# --- Rate limiter for Dataset eval hook ---
# Max 1 evaluation trigger per second across all Dataset creations.
_eval_rate_lock = threading.Lock()
_eval_last_run: float = 0.0


@receiver(pre_save, sender="model_hub.PromptVersion")
def capture_prompt_version_old_state(sender, instance, **kwargs):
    """Capture old state before save for comparison in post_save."""
    if instance.pk:
        # Only capture if this is an update (not a new creation)
        from model_hub.models.run_prompt import PromptVersion

        try:
            old_instance = PromptVersion.objects.get(pk=instance.pk)
            # Store old state in thread-local storage
            _thread_locals.prompt_version_old_state = {
                "variable_names": old_instance.variable_names,
                "prompt_config_snapshot": old_instance.prompt_config_snapshot,
            }
        except PromptVersion.DoesNotExist:
            _thread_locals.prompt_version_old_state = None
    else:
        _thread_locals.prompt_version_old_state = None


@receiver(post_save, sender="model_hub.PromptVersion")
def sync_agent_playground_on_prompt_change(sender, instance, created, **kwargs):
    """
    Auto-sync agent playground nodes when PromptVersion changes.

    Triggers sync when:
    - Variables change ({{var}} additions/removals in messages)
    - response_format changes (string/json/json_schema)

    Does NOT trigger on:
    - Message content changes (not stored at node level)
    - is_draft or is_default flag changes
    - Metadata or evaluation config changes
    """
    # Skip if this is a new creation (no previous state to compare)
    if created:
        _thread_locals.prompt_version_old_state = None
        return

    # Check if relevant fields changed
    if not _has_relevant_changes(instance):
        _thread_locals.prompt_version_old_state = None
        return

    # Import here to avoid circular imports
    from agent_playground.services.prompt_sync import sync_nodes_for_prompt_version

    try:
        sync_nodes_for_prompt_version(instance)
        logger.info(
            "agent_playground_sync_completed",
            prompt_version_id=str(instance.id),
            prompt_template_id=str(instance.original_template_id),
        )
    except Exception as e:
        logger.error(
            "agent_playground_sync_failed",
            prompt_version_id=str(instance.id),
            error=str(e),
            exc_info=True,
        )
        # Don't raise - allow prompt save to succeed even if sync fails
    finally:
        # Clean up thread-local storage
        _thread_locals.prompt_version_old_state = None


def _has_relevant_changes(prompt_version) -> bool:
    """
    Check if PromptVersion has changes requiring agent playground sync.

    Compares current state with previous state captured in pre_save.
    Returns True if variables or response_format changed.
    """
    # Get old state from thread-local storage (captured in pre_save)
    old_state = getattr(_thread_locals, "prompt_version_old_state", None)
    if not old_state:
        return False

    def _variable_keys(value):
        if isinstance(value, dict):
            return set(value.keys())
        if isinstance(value, (list, tuple, set)):
            return set(value)
        return set()

    old_vars = _variable_keys(old_state.get("variable_names"))
    new_vars = _variable_keys(prompt_version.variable_names)

    if old_vars != new_vars:
        logger.info(
            "variables_changed",
            prompt_version_id=str(prompt_version.id),
            added=list(new_vars - old_vars),
            removed=list(old_vars - new_vars),
        )
        return True

    # Check if response_format changed
    def _response_format(config):
        if not isinstance(config, dict):
            return None
        configuration = config.get("configuration")
        if not isinstance(configuration, dict):
            return None
        return configuration.get("response_format")

    old_format = _response_format(old_state.get("prompt_config_snapshot"))
    new_format = _response_format(prompt_version.prompt_config_snapshot)

    if old_format != new_format:
        logger.info(
            "response_format_changed",
            prompt_version_id=str(prompt_version.id),
            old_format=old_format,
            new_format=new_format,
        )
        return True

    return False


@receiver(pre_save, sender="model_hub.Dataset")
def cascade_soft_delete_on_dataset_deletion(sender, instance, **kwargs):
    """
    When a Dataset is soft-deleted, cascade soft-delete all experiments
    that use it as their source dataset (along with their EDTs, EPCs, EACs,
    and snapshot datasets).
    """
    if not instance.pk:
        return

    from model_hub.models.develop_dataset import Dataset

    try:
        old_instance = Dataset.all_objects.get(pk=instance.pk)
    except Dataset.DoesNotExist:
        return

    if not old_instance.deleted and instance.deleted:
        _cascade_soft_delete_dataset_experiments(instance)


def _cascade_soft_delete_dataset_experiments(dataset):
    """Soft-delete experiments and their children when source dataset is deleted."""
    from django.utils import timezone

    from model_hub.models.experiments import (
        ExperimentAgentConfig,
        ExperimentDatasetTable,
        ExperimentPromptConfig,
        ExperimentsTable,
    )

    now = timezone.now()

    experiments = ExperimentsTable.all_objects.filter(dataset=dataset, deleted=False)

    for exp in experiments:
        # Soft-delete EDTs, EPCs, and EACs
        edts = ExperimentDatasetTable.all_objects.filter(experiment=exp, deleted=False)
        edt_ids = list(edts.values_list("id", flat=True))

        ExperimentPromptConfig.all_objects.filter(
            experiment_dataset_id__in=edt_ids, deleted=False
        ).update(deleted=True, deleted_at=now)

        ExperimentAgentConfig.all_objects.filter(
            experiment_dataset_id__in=edt_ids, deleted=False
        ).update(deleted=True, deleted_at=now)

        edts.update(deleted=True, deleted_at=now)

        # Soft-delete snapshot dataset (triggers this signal recursively
        # to clean up the snapshot's experiments if any)
        if exp.snapshot_dataset_id:
            snapshot = exp.snapshot_dataset
            if snapshot and not snapshot.deleted:
                snapshot.deleted = True
                snapshot.deleted_at = now
                snapshot.save(update_fields=["deleted", "deleted_at"])

    experiments.update(deleted=True, deleted_at=now)

    logger.info(
        "dataset_soft_delete_cascaded",
        dataset_id=str(dataset.id),
        experiments_deleted=experiments.count(),
    )


@receiver(pre_save, sender="model_hub.PromptTemplate")
def cascade_soft_delete_on_prompt_template_deletion(sender, instance, **kwargs):
    """
    When a PromptTemplate is soft-deleted, cascade soft-delete all linked agent playground nodes.
    """
    if not instance.pk:
        return  # New object, not an update

    try:
        old_instance = sender.no_workspace_objects.get(pk=instance.pk)
    except sender.DoesNotExist:
        return  # Should not happen, but be defensive

    # Check if this is a soft delete operation
    if not old_instance.deleted and instance.deleted:
        # Soft delete is happening - cascade to nodes
        from agent_playground.models.prompt_template_node import PromptTemplateNode
        from agent_playground.services.node_crud import cascade_soft_delete_node

        # Find all PromptTemplateNodes linked to this template
        ptn_records = PromptTemplateNode.no_workspace_objects.filter(
            prompt_template=instance, deleted=False
        ).select_related("node")

        nodes_cascaded = 0
        for ptn in ptn_records:
            # Cascade soft delete the node (which will also delete the PTN)
            cascade_soft_delete_node(ptn.node)
            nodes_cascaded += 1

        if nodes_cascaded > 0:
            logger.info(
                "prompt_template_soft_delete_cascaded",
                prompt_template_id=str(instance.id),
                nodes_deleted=nodes_cascaded,
            )


@receiver(pre_save, sender="model_hub.PromptVersion")
def cascade_soft_delete_on_prompt_version_deletion(sender, instance, **kwargs):
    """
    When a PromptVersion is soft-deleted, cascade soft-delete all linked agent playground nodes.
    """
    if not instance.pk:
        return  # New object, not an update

    # Import PromptVersion to get the actual model class with managers
    from model_hub.models.run_prompt import PromptVersion

    try:
        # PromptVersion doesn't have a workspace field, so use all_objects
        old_instance = PromptVersion.all_objects.get(pk=instance.pk)
    except PromptVersion.DoesNotExist:
        return

    # Check if this is a soft delete operation
    if not old_instance.deleted and instance.deleted:
        # Soft delete is happening - cascade to nodes
        from agent_playground.models.prompt_template_node import PromptTemplateNode
        from agent_playground.services.node_crud import cascade_soft_delete_node

        # Find all PromptTemplateNodes linked to this version
        ptn_records = PromptTemplateNode.no_workspace_objects.filter(
            prompt_version=instance, deleted=False
        ).select_related("node")

        nodes_cascaded = 0
        for ptn in ptn_records:
            # Cascade soft delete the node (which will also delete the PTN)
            cascade_soft_delete_node(ptn.node)
            nodes_cascaded += 1

        if nodes_cascaded > 0:
            logger.info(
                "prompt_version_soft_delete_cascaded",
                prompt_version_id=str(instance.id),
                nodes_deleted=nodes_cascaded,
            )


# ──────────────────────────────────────────────────────────────────────
# Evaluation hook — auto-run evaluations when datasets are created
# ──────────────────────────────────────────────────────────────────────

_DATASET_EVAL_HOOK_ENABLED = True  # Toggle for operational control


def _evaluation_rate_limited() -> bool:
    """Return True if the rate limit is exceeded, False if allowed."""
    global _eval_last_run
    with _eval_rate_lock:
        now = time.time()
        if now - _eval_last_run < 1.0:
            return True  # Rate-limited
        _eval_last_run = now
        return False


def _trigger_dataset_evals(dataset) -> None:
    """
    Schedule evaluations for a newly created dataset.

    Looks up UserEvalMetric records associated with the dataset and
    fires off evaluation runs asynchronously.  Falls back to dataset-level
    eval_reasons if no UserEvalMetric records are found.

    This function is intentionally fire-and-forget — failures are logged
    but never propagated back to the Dataset save path.
    """
    from model_hub.models.evals_metric import UserEvalMetric

    try:
        user_eval_metrics = list(
            UserEvalMetric.objects.filter(
                dataset=dataset, deleted=False
            ).select_related("template", "organization")
        )
    except Exception:
        logger.exception(
            "eval_hook_failed_to_query_metrics",
            dataset_id=str(dataset.id),
        )
        return

    if not user_eval_metrics:
        logger.debug(
            "eval_hook_no_metrics_found",
            dataset_id=str(dataset.id),
            reason="No UserEvalMetric rows linked to this dataset yet.",
        )
        return

    triggered = 0
    for uem in user_eval_metrics:
        try:
            _run_single_eval_metric(uem, dataset)
            triggered += 1
        except Exception:
            logger.exception(
                "eval_hook_single_failed",
                dataset_id=str(dataset.id),
                metric_id=str(uem.id),
            )

    if triggered:
        logger.info(
            "eval_hook_triggered",
            dataset_id=str(dataset.id),
            dataset_name=dataset.name,
            metrics_triggered=triggered,
        )


def _run_single_eval_metric(user_eval_metric, dataset) -> None:
    """Fire a single evaluation for one UserEvalMetric on a dataset.

    Delegates to the shared ``run_eval_func`` for consistency with
    the rest of the evaluation pipeline (playground, experiment runner,
    composite runner, etc.).
    """
    from model_hub.views.utils.evals import run_eval_func

    template = user_eval_metric.template
    org = user_eval_metric.organization
    config = user_eval_metric.config or {}

    # Build mappings from the metric config — the dataset-level context
    # (columns, rows) is resolved by the runner when it encounters
    # `{{column.<id>}}` / `{{row.<id>}}` templates.
    mappings = config.get("mapping", {}).copy()

    run_eval_func(
        config=config,
        mappings=mappings,
        template=template,
        org=org,
        model=config.get("model") or "turing_large",
        workspace=getattr(dataset, "workspace", None),
        input_data_types=config.get("input_data_types", {}),
        kb_id=config.get("kb_id", None),
        error_localizer=config.get("error_localizer", False),
    )


@receiver(post_save, sender="model_hub.Dataset")
def dataset_eval_hook(sender, instance, created, **kwargs):
    """
    Post-save signal handler that triggers evaluations when new Datasets
    are created.

    Safety guarantees:
    1. Only fires on creation (``created=True``) — not on updates.
    2. Rate-limited to 1 trigger / second across all Dataset creations.
    3. Exceptions are caught and logged; Dataset saves are never blocked.
    4. A global toggle (``_DATASET_EVAL_HOOK_ENABLED``) allows disabling
       the hook without a code deploy.
    """
    if not created:
        return

    if not _DATASET_EVAL_HOOK_ENABLED:
        return

    if _evaluation_rate_limited():
        logger.info(
            "eval_hook_rate_limited",
            dataset_id=str(instance.id),
            dataset_name=instance.name,
        )
        return

    try:
        _trigger_dataset_evals(instance)
    except Exception:
        logger.exception(
            "eval_hook_unexpected_failure",
            dataset_id=str(instance.id),
            dataset_name=instance.name,
        )
