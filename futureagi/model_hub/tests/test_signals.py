"""Tests for model_hub signals."""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.django_db


class TestPromptVersionSignals:
    """Test Django signals for PromptVersion model."""

    @patch("agent_playground.services.prompt_sync.sync_nodes_for_prompt_version")
    def test_signal_triggered_on_variable_change(self, mock_sync, prompt_version):
        """Test that signal fires when variables change."""
        # Initial save
        prompt_version.prompt_config_snapshot = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi {{name}}"}],
                }
            ],
            "configuration": {},
        }
        prompt_version.variable_names = {"name": []}
        prompt_version.save()

        # Clear mock calls from initial save
        mock_sync.reset_mock()

        # Update with new variable
        prompt_version.variable_names = {"name": [], "city": []}
        prompt_version.save()

        # Verify sync was called
        assert mock_sync.called
        assert mock_sync.call_args[0][0] == prompt_version

    @patch("agent_playground.services.prompt_sync.sync_nodes_for_prompt_version")
    def test_signal_not_triggered_on_metadata_change(self, mock_sync, prompt_version):
        """Test that signal doesn't fire for irrelevant changes."""
        # Set initial state
        prompt_version.prompt_config_snapshot = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi {{name}}"}],
                }
            ],
            "configuration": {},
        }
        prompt_version.variable_names = {"name": []}
        prompt_version.save()

        # Clear mock calls from initial save
        mock_sync.reset_mock()

        # Update irrelevant field (metadata)
        prompt_version.metadata = {"key": "value"}
        prompt_version.save()

        # Verify sync was NOT called
        assert not mock_sync.called

    @patch("agent_playground.services.prompt_sync.sync_nodes_for_prompt_version")
    def test_signal_triggered_on_response_format_change(
        self, mock_sync, prompt_version
    ):
        """Test that signal fires when response_format changes."""
        # Initial save
        prompt_version.prompt_config_snapshot = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi"}],
                }
            ],
            "configuration": {"response_format": {"type": "text"}},
        }
        prompt_version.variable_names = {}
        prompt_version.save()

        # Clear mock calls from initial save
        mock_sync.reset_mock()

        # Update response_format
        prompt_version.prompt_config_snapshot = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi"}],
                }
            ],
            "configuration": {"response_format": {"type": "json_object"}},
        }
        prompt_version.save()

        # Verify sync was called
        assert mock_sync.called
        assert mock_sync.call_args[0][0] == prompt_version

    @patch("agent_playground.services.prompt_sync.sync_nodes_for_prompt_version")
    def test_signal_not_triggered_on_creation(self, mock_sync, organization, workspace):
        """Test that signal doesn't fire when PromptVersion is first created."""
        from model_hub.models.run_prompt import PromptTemplate, PromptVersion

        # Create new PromptTemplate
        template = PromptTemplate.objects.create(
            name="Test Template",
            organization=organization,
            workspace=workspace,
        )

        # Create new PromptVersion (created=True)
        new_version = PromptVersion.objects.create(
            original_template=template,
            prompt_config_snapshot={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hi {{name}}"}],
                    }
                ],
                "configuration": {"response_format": {"type": "text"}},
            },
            variable_names={"name": []},
        )

        # Verify sync was NOT called (because created=True)
        assert not mock_sync.called


class TestDatasetEvalHookSignals:
    """Test Django signals for Dataset eval hook."""

    @patch("model_hub.signals._trigger_dataset_evals")
    def test_eval_hook_fires_on_creation(
        self, mock_trigger, organization, workspace, user
    ):
        """Test that eval hook fires when a new Dataset is created."""
        from model_hub.models.develop_dataset import Dataset
        from model_hub.models.choices import DatasetSourceChoices, ModelTypes

        dataset = Dataset.objects.create(
            name="eval-hook-test-dataset",
            organization=organization,
            user=user,
            source=DatasetSourceChoices.BUILD.value,
            model_type=ModelTypes.GENERATIVE_LLM.value,
            workspace=workspace,
        )

        # Signal fires on creation (post_save with created=True)
        assert mock_trigger.called
        assert mock_trigger.call_args[0][0] == dataset

    @patch("model_hub.signals._trigger_dataset_evals")
    def test_eval_hook_does_not_fire_on_update(
        self, mock_trigger, organization, workspace, user
    ):
        """Test that eval hook does NOT fire when a Dataset is updated."""
        from model_hub.models.develop_dataset import Dataset
        from model_hub.models.choices import DatasetSourceChoices, ModelTypes

        dataset = Dataset.objects.create(
            name="eval-hook-update-test",
            organization=organization,
            user=user,
            source=DatasetSourceChoices.BUILD.value,
            model_type=ModelTypes.GENERATIVE_LLM.value,
            workspace=workspace,
        )

        # Clear call count from creation
        mock_trigger.reset_mock()

        # Update the dataset
        dataset.name = "updated-name"
        dataset.save()

        # Signal should NOT fire on update (created=False)
        assert not mock_trigger.called

    @patch("model_hub.signals._trigger_dataset_evals")
    def test_eval_hook_rate_limiting(
        self, mock_trigger, organization, workspace, user
    ):
        """Test that rate limiting prevents rapid-fire evaluations."""
        import model_hub.signals as signals_module

        from model_hub.models.develop_dataset import Dataset
        from model_hub.models.choices import DatasetSourceChoices, ModelTypes

        # Force rate limit state to "just ran" so next call is rate-limited
        with signals_module._eval_rate_lock:
            signals_module._eval_last_run = time_now()

        dataset = Dataset.objects.create(
            name="rate-limited-dataset",
            organization=organization,
            user=user,
            source=DatasetSourceChoices.BUILD.value,
            model_type=ModelTypes.GENERATIVE_LLM.value,
            workspace=workspace,
        )

        # Trigger should be skipped due to rate limiting
        assert not mock_trigger.called

        # Reset rate limit and verify it works again
        with signals_module._eval_rate_lock:
            signals_module._eval_last_run = 0.0

        dataset2 = Dataset.objects.create(
            name="rate-limited-dataset-2",
            organization=organization,
            user=user,
            source=DatasetSourceChoices.BUILD.value,
            model_type=ModelTypes.GENERATIVE_LLM.value,
            workspace=workspace,
        )

        assert mock_trigger.called

    @patch("model_hub.signals._trigger_dataset_evals")
    def test_eval_hook_survives_exceptions(
        self, mock_trigger, organization, workspace, user
    ):
        """Test that eval hook failures do not block Dataset creation."""
        from model_hub.models.develop_dataset import Dataset
        from model_hub.models.choices import DatasetSourceChoices, ModelTypes

        mock_trigger.side_effect = RuntimeError("Simulated eval hook failure")

        # Dataset creation should still succeed even if eval hook fails
        dataset = Dataset.objects.create(
            name="exception-safe-dataset",
            organization=organization,
            user=user,
            source=DatasetSourceChoices.BUILD.value,
            model_type=ModelTypes.GENERATIVE_LLM.value,
            workspace=workspace,
        )

        assert dataset.pk is not None
        assert dataset.name == "exception-safe-dataset"

    @patch("model_hub.signals._trigger_dataset_evals")
    def test_eval_hook_disabled_toggle(
        self, mock_trigger, organization, workspace, user
    ):
        """Test that the global toggle can disable the eval hook."""
        import model_hub.signals as signals_module

        from model_hub.models.develop_dataset import Dataset
        from model_hub.models.choices import DatasetSourceChoices, ModelTypes

        # Disable the hook
        signals_module._DATASET_EVAL_HOOK_ENABLED = False

        try:
            Dataset.objects.create(
                name="disabled-hook-dataset",
                organization=organization,
                user=user,
                source=DatasetSourceChoices.BUILD.value,
                model_type=ModelTypes.GENERATIVE_LLM.value,
                workspace=workspace,
            )

            assert not mock_trigger.called
        finally:
            # Restore
            signals_module._DATASET_EVAL_HOOK_ENABLED = True

    @patch("model_hub.signals._run_single_eval_metric")
    def test_trigger_dataset_evals_with_metrics(
        self, mock_run_single, organization, workspace, user
    ):
        """Test that _trigger_dataset_evals triggers UserEvalMetrics."""
        import time as time_module

        import model_hub.signals as signals_module

        from model_hub.models.develop_dataset import Column, Dataset
        from model_hub.models.choices import (
            DataTypeChoices,
            DatasetSourceChoices,
            EvalTemplateType,
            ModelTypes,
            SourceChoices,
        )
        from model_hub.models.evals_metric import EvalTemplate, UserEvalMetric

        # Create dataset
        dataset = Dataset.objects.create(
            name="trigger-test-dataset",
            organization=organization,
            user=user,
            source=DatasetSourceChoices.BUILD.value,
            model_type=ModelTypes.GENERATIVE_LLM.value,
            workspace=workspace,
        )

        # Create a column (UserEvalMetric requires dataset with columns)
        col = Column.objects.create(
            dataset=dataset,
            name="input_col",
            data_type=DataTypeChoices.TEXT.value,
            source=SourceChoices.OTHERS.value,
        )

        # Create eval template
        eval_template = EvalTemplate.objects.create(
            name="Test Eval Template",
            organization=organization,
            workspace=workspace,
            config={"output": "bool", "eval_type_id": "example"},
            creator=user,
        )

        # Create UserEvalMetric linked to dataset
        uem = UserEvalMetric.objects.create(
            name="test-metric",
            dataset=dataset,
            template=eval_template,
            organization=organization,
            workspace=workspace,
            user=user,
            config={"mapping": {"input": str(col.id)}, "model": "turing_large"},
        )

        # Reset rate limiter
        with signals_module._eval_rate_lock:
            signals_module._eval_last_run = 0.0

        # Trigger
        signals_module._trigger_dataset_evals(dataset)

        # Verify _run_single_eval_metric was called with the UserEvalMetric
        assert mock_run_single.called
        call_args = mock_run_single.call_args[0]
        assert call_args[0] == uem
        assert call_args[1] == dataset

    @patch("model_hub.signals._run_single_eval_metric")
    def test_trigger_dataset_evals_no_metrics(
        self, mock_run_single, organization, workspace, user
    ):
        """Test that _trigger_dataset_evals is a no-op when no metrics exist."""
        import model_hub.signals as signals_module

        from model_hub.models.develop_dataset import Dataset
        from model_hub.models.choices import DatasetSourceChoices, ModelTypes

        dataset = Dataset.objects.create(
            name="no-metrics-dataset",
            organization=organization,
            user=user,
            source=DatasetSourceChoices.BUILD.value,
            model_type=ModelTypes.GENERATIVE_LLM.value,
            workspace=workspace,
        )

        # No UserEvalMetric exists — should be a clean no-op
        signals_module._trigger_dataset_evals(dataset)

        assert not mock_run_single.called


def time_now():
    import time
    return time.time()
