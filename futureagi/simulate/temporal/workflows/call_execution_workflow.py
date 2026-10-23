"""
CallExecutionWorkflow - Individual call execution lifecycle.

Manages the complete lifecycle of a single simulated call:
1. PREPARING: Fetch call data from DB, build system prompt
2. ACQUIRING: Request rate limit slot from dispatcher, wait for grant
3. CALLING: Initiate call with voice provider (VAPI/LiveKit)
4. MONITORING: Poll provider until call ends (up to 30 min)
5. STORING: Fetch and persist transcript, recording, costs
6. EVALUATING: Run configured evaluations on call output
7. DOWNSTREAM_TASKS: Run post-call processing (webhooks, metrics, etc.)
8. CLEANUP: Release rate limit slot, release phone, signal parent

Design:
- Signal-driven slot allocation from CallDispatcherWorkflow
- Graceful cancellation support
- Fire-and-forget error reporting on unexpected failures
- Each phase can fail-fast with proper cleanup
"""

import asyncio
from datetime import timedelta
from typing import Optional

from temporalio import workflow

from simulate.temporal.constants import (
    MAX_CALL_DURATION_SECONDS,
    MIN_DURATION_SECONDS_WITHOUT_TRANSCRIPT,
    QUEUE_L,
    QUEUE_S,
    QUEUE_XL,
)
from simulate.temporal.retry_policies import (
    DB_RETRY_POLICY,
    EVAL_RETRY_POLICY,
    PROVIDER_RETRY_POLICY,
    SIGNAL_RETRY_POLICY,
)
from simulate.temporal.signals import (
    SIGNAL_CANCEL_CALL,
    SIGNAL_SLOT_GRANTED,
)
from simulate.temporal.types.activities import (
    AcquirePhoneInput,
    AcquirePhoneOutput,
    CalculateConversationMetricsInput,
    CalculateVoiceCSATInput,
    CalculateVoiceCSATOutput,
    CheckBalanceInput,
    CheckBalanceOutput,
    DownstreamTasksInput,
    FetchAndPersistCallResultInput,
    FetchAndPersistCallResultOutput,
    InitiateCallInput,
    InitiateCallOutput,
    MonitorCallInput,
    MonitorCallOutput,
    ReleasePhoneInput,
    ReleaseSlotInput,
    ReportErrorInput,
    RequestSlotInput,
    RunSimulateEvaluationsInput,
    RunSimulateEvaluationsOutput,
    RunToolCallEvaluationInput,
    RunToolCallEvaluationOutput,
    SignalCallAnalyzingInput,
    SignalCallCompleteInput,
    UpdateCallStatusInput,
    UpdateMetricsInput,
)
from simulate.temporal.types.call_execution import (
    CallExecutionInput,
    CallExecutionOutput,
    CallStatus,
)
from simulate.temporal.types.activities import PrepareCallInput, PrepareCallOutput


@workflow.defn
class CallExecutionWorkflow:
    """
    Individual call execution workflow.

    Manages the lifecycle of a single simulated call from preparation
    through evaluation and cleanup. Communicates with the parent
    TestExecutionWorkflow via signals for progress tracking.

    Phases:
    1. PREPARING: Load call config from DB
    2. ACQUIRING: Get rate limit slot from dispatcher
    3. CALLING: Create call with provider
    4. MONITORING: Wait for call completion
    5. STORING: Persist call results
    6. EVALUATING: Run evals + CSAT + metrics
    7. DOWNSTREAM_TASKS: Post-call processing
    8. CLEANUP: Release slots, signal parent
    """

    def __init__(self):
        self._status = "PENDING"
        self._call_id: Optional[str] = None
        self._org_id: Optional[str] = None
        self._workspace_id: Optional[str] = None
        self._test_workflow_id: Optional[str] = None
        self._test_execution_id: Optional[str] = None
        self._eval_only: bool = False

        # Call execution state
        self._provider_call_id: Optional[str] = None
        self._phone_id: Optional[str] = None
        self._call_duration_seconds: Optional[int] = None
        self._call_status: Optional[str] = None
        self._call_error: Optional[str] = None

        # Slot grant coordination
        self._slot_granted = False

    @workflow.run
    async def run(self, input: CallExecutionInput) -> CallExecutionOutput:
        """Main workflow execution."""
        self._call_id = input.call_id
        self._org_id = input.org_id
        self._workspace_id = input.workspace_id
        self._test_workflow_id = input.test_workflow_id
        self._test_execution_id = input.test_execution_id
        self._eval_only = input.eval_only

        workflow.logger.info(
            f"CallExecutionWorkflow started: call_id={self._call_id}, "
            f"eval_only={self._eval_only}"
        )

        try:
            # ========================================
            # PHASE 1: PREPARING
            # ========================================
            self._status = "PREPARING"

            prep_result = await workflow.execute_activity(
                "prepare_call",
                PrepareCallInput(
                    call_id=self._call_id,
                    workspace_id=self._workspace_id,
                ),
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=DB_RETRY_POLICY,
                task_queue=QUEUE_L,
                result_type=PrepareCallOutput,
            )

            if prep_result.error:
                return await self._fail(f"Call preparation failed: {prep_result.error}")

            # Update call status to show preparation is done
            await workflow.execute_activity(
                "update_call_status",
                UpdateCallStatusInput(
                    call_id=self._call_id,
                    status="REGISTERED",
                    test_execution_id=self._test_execution_id,
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=DB_RETRY_POLICY,
                task_queue=QUEUE_S,
            )

            # Eval-only mode: skip call execution, go straight to evaluations
            if self._eval_only:
                workflow.logger.info(
                    f"Eval-only mode: skipping call execution for {self._call_id}"
                )
                return await self._run_eval_only_flow(prep_result)

            # ========================================
            # PHASE 2: BALANCE CHECK
            # ========================================
            balance_result = await workflow.execute_activity(
                "check_call_balance",
                CheckBalanceInput(
                    org_id=self._org_id,
                    estimated_duration_minutes=prep_result.max_duration_minutes,
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=DB_RETRY_POLICY,
                task_queue=QUEUE_S,
                result_type=CheckBalanceOutput,
            )

            if not balance_result.sufficient:
                return await self._fail(
                    f"Insufficient balance: current={balance_result.current_balance}, "
                    f"estimated_cost={balance_result.estimated_cost}"
                )

            # ========================================
            # PHASE 3: PHONE ACQUISITION (outbound only)
            # ========================================
            phone_result = None
            if prep_result.is_outbound:
                phone_result = await workflow.execute_activity(
                    "acquire_phone_number",
                    AcquirePhoneInput(
                        call_id=self._call_id,
                        call_direction="outbound",
                    ),
                    start_to_close_timeout=timedelta(minutes=10),
                    heartbeat_timeout=timedelta(minutes=1),
                    retry_policy=DB_RETRY_POLICY,
                    task_queue=QUEUE_S,
                    result_type=AcquirePhoneOutput,
                )

                if not phone_result.success:
                    return await self._fail(
                        f"Failed to acquire phone: {phone_result.error}"
                    )
                self._phone_id = phone_result.phone_id

            # ========================================
            # PHASE 4: ACQUIRING (rate limit slot)
            # ========================================
            self._status = "ACQUIRING"

            await workflow.execute_activity(
                "request_call_slot",
                RequestSlotInput(
                    workflow_id=workflow.info().workflow_id,
                    call_id=self._call_id,
                    org_id=self._org_id,
                ),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=SIGNAL_RETRY_POLICY,
                task_queue=QUEUE_S,
            )

            # Wait for slot grant from dispatcher
            slot_granted = await self._wait_for_slot()
            if not slot_granted:
                return await self._fail("Cancelled while waiting for slot")

            await workflow.execute_activity(
                "update_call_status",
                UpdateCallStatusInput(
                    call_id=self._call_id,
                    status="ONGOING",
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=DB_RETRY_POLICY,
                task_queue=QUEUE_S,
            )

            # ========================================
            # PHASE 5: CALLING
            # ========================================
            self._status = "CALLING"

            # Build call data dict from prepare output
            call_data = prep_result.call_data.copy()
            call_data.update({
                "is_outbound": prep_result.is_outbound,
                "provider": prep_result.provider,
            })

            initiate_result = await workflow.execute_activity(
                "initiate_call",
                InitiateCallInput(
                    call_id=self._call_id,
                    call_data=call_data,
                    system_prompt=prep_result.system_prompt,
                    voice_settings=prep_result.voice_settings,
                    metadata=prep_result.metadata,
                    provider=prep_result.provider,
                    # Outbound call fields
                    phone_number=(
                        prep_result.customer_phone_number
                        if prep_result.is_outbound
                        else None
                    ),
                    provider_phone_id=(
                        getattr(phone_result, "provider_phone_id", None)
                        if prep_result.is_outbound and phone_result
                        else None
                    ),
                    user_api_key=prep_result.client_api_key,
                    user_assistant_id=prep_result.client_assistant_id,
                    user_phone_number=prep_result.client_phone_number,
                    client_provider=prep_result.client_provider,
                    connection_type=prep_result.connection_type,
                ),
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=PROVIDER_RETRY_POLICY,
                task_queue=QUEUE_L,
                result_type=InitiateCallOutput,
            )

            if not initiate_result.success:
                return await self._fail(
                    f"Call initiation failed: {initiate_result.error}"
                )

            self._provider_call_id = initiate_result.provider_call_id

            # ========================================
            # PHASE 6: MONITORING
            # ========================================
            self._status = "MONITORING"

            monitor_result = await workflow.execute_activity(
                "monitor_call_until_complete",
                MonitorCallInput(
                    call_id=self._call_id,
                    provider_call_id=self._provider_call_id,
                    call_type=(
                        "inbound"
                        if not prep_result.is_outbound
                        else "outbound"
                    ),
                    provider=prep_result.provider,
                    client_provider=prep_result.client_provider,
                    provider_config=prep_result.provider_config,
                    max_duration_seconds=MAX_CALL_DURATION_SECONDS,
                ),
                start_to_close_timeout=timedelta(minutes=35),
                heartbeat_timeout=timedelta(minutes=1),
                retry_policy=PROVIDER_RETRY_POLICY,
                task_queue=QUEUE_L,
                result_type=MonitorCallOutput,
            )

            if not monitor_result.success:
                workflow.logger.warning(
                    f"Call monitoring failed: {monitor_result.error}"
                )
                # Don't fail the entire workflow - continue to store what we have

            self._call_duration_seconds = monitor_result.duration_seconds
            self._call_status = monitor_result.status

            # ========================================
            # PHASE 7: STORING
            # ========================================
            self._status = "STORING"

            persist_result = await workflow.execute_activity(
                "fetch_and_persist_call_result",
                FetchAndPersistCallResultInput(
                    call_id=self._call_id,
                    status=monitor_result.status or "ended",
                    provider_call_id=self._provider_call_id,
                    provider=prep_result.provider,
                    call_type=(
                        "inbound"
                        if not prep_result.is_outbound
                        else "outbound"
                    ),
                    client_provider=prep_result.client_provider,
                    duration_seconds=self._call_duration_seconds,
                    end_reason=monitor_result.end_reason,
                    provider_data=initiate_result.provider_data,
                    provider_config=prep_result.provider_config,
                ),
                start_to_close_timeout=timedelta(minutes=5),
                heartbeat_timeout=timedelta(minutes=1),
                retry_policy=DB_RETRY_POLICY,
                task_queue=QUEUE_L,
                result_type=FetchAndPersistCallResultOutput,
            )

            if not persist_result.success:
                workflow.logger.warning(
                    f"Failed to persist call results: {persist_result.error}"
                )

            # ========================================
            # PHASE 8: EVALUATING
            # ========================================
            self._status = "EVALUATING"

            # Signal parent that we're now analyzing (evaluations starting)
            await workflow.execute_activity(
                "signal_call_analyzing",
                SignalCallAnalyzingInput(
                    workflow_id=self._test_workflow_id,
                    call_id=self._call_id,
                ),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=SIGNAL_RETRY_POLICY,
                task_queue=QUEUE_S,
            )

            # Run standard simulate evaluations
            eval_ok = await self._run_evaluations(prep_result)

            # Run CSAT evaluation if recording is available
            await self._run_csat_evaluation(persist_result)

            # Calculate conversation metrics
            await self._calculate_conversation_metrics(prep_result)

            # Run tool call evaluation if enabled
            await self._run_tool_evaluation(prep_result)

            # Update metrics
            if eval_ok:
                await workflow.execute_activity(
                    "update_call_metrics",
                    UpdateMetricsInput(
                        call_id=self._call_id,
                        test_execution_id=self._test_execution_id,
                    ),
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=DB_RETRY_POLICY,
                    task_queue=QUEUE_S,
                )

            # ========================================
            # PHASE 9: DOWNSTREAM TASKS
            # ========================================
            self._status = "DOWNSTREAM_TASKS"

            await workflow.execute_activity(
                "run_downstream_tasks",
                DownstreamTasksInput(
                    call_id=self._call_id,
                    test_execution_id=self._test_execution_id,
                    call_status=self._call_status or "ended",
                ),
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=DB_RETRY_POLICY,
                task_queue=QUEUE_L,
            )

            # ========================================
            # PHASE 10: COMPLETION
            # ========================================
            return await self._complete()

        except asyncio.CancelledError:
            workflow.logger.info(
                f"CallExecutionWorkflow cancelled: call_id={self._call_id}"
            )
            return await self._handle_cancellation()

        except Exception as e:
            workflow.logger.error(
                f"CallExecutionWorkflow failed: {str(e)}"
            )
            # Fire-and-forget error report
            workflow.start_activity(
                "report_workflow_error",
                ReportErrorInput(
                    workflow_name="CallExecutionWorkflow",
                    workflow_id=workflow.info().workflow_id,
                    error_message=str(e),
                    error_type=type(e).__name__,
                    context={
                        "call_id": self._call_id,
                        "test_execution_id": self._test_execution_id,
                        "status": self._status,
                    },
                ),
                start_to_close_timeout=timedelta(seconds=10),
                task_queue=QUEUE_S,
            )
            return await self._fail(str(e))

    # ========================================
    # SIGNAL HANDLERS
    # ========================================

    @workflow.signal
    async def slot_granted(self) -> None:
        """Signal from CallDispatcherWorkflow when slot is granted."""
        self._slot_granted = True
        workflow.logger.info(
            f"Slot granted for call_id={self._call_id}"
        )

    @workflow.signal
    async def cancel(self) -> None:
        """Cooperative cancel signal for immediate cancellation.

        Sets the cancelled flag so waiting loops can exit early.
        No return value — signals are 1-way (wf → signal handler).
        """
        self._slot_granted = True  # Unblock _wait_for_slot
        workflow.logger.info(
            f"Cancel signal received for call_id={self._call_id}"
        )

    # ========================================
    # QUERIES
    # ========================================

    @workflow.query
    def get_status(self) -> CallStatus:
        """Query current workflow status."""
        return CallStatus(
            status=self._status,
            provider_call_id=self._provider_call_id,
            phase_details=self._call_status,
        )

    # ========================================
    # HELPER METHODS
    # ========================================

    async def _wait_for_slot(self, timeout_seconds: int = 300) -> bool:
        """Wait for SLOT_GRANTED signal from dispatcher.

        Returns True if slot granted, False if timed out or cancelled.
        """
        deadline = workflow.time() + timeout_seconds

        while not self._slot_granted:
            if workflow.time() >= deadline:
                workflow.logger.warning(
                    f"Slot grant timeout for call_id={self._call_id}"
                )
                return False

            await workflow.sleep(2)

        return True

    async def _run_eval_only_flow(
        self, prep_result: PrepareCallOutput
    ) -> CallExecutionOutput:
        """Run evaluations on existing call data without executing a new call."""
        self._status = "EVALUATING"

        # Signal parent that we're analyzing
        await workflow.execute_activity(
            "signal_call_analyzing",
            SignalCallAnalyzingInput(
                workflow_id=self._test_workflow_id,
                call_id=self._call_id,
            ),
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=SIGNAL_RETRY_POLICY,
            task_queue=QUEUE_S,
        )

        await self._run_evaluations(prep_result)
        await self._run_tool_evaluation(prep_result)

        return await self._complete()

    async def _run_evaluations(
        self, prep_result: PrepareCallOutput
    ) -> bool:
        """Run standard simulate evaluations. Returns True if successful."""
        try:
            eval_result = await workflow.execute_activity(
                "run_simulate_evaluations",
                RunSimulateEvaluationsInput(
                    call_execution_id=self._call_id,
                ),
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(minutes=1),
                retry_policy=EVAL_RETRY_POLICY,
                task_queue=QUEUE_XL,
                result_type=RunSimulateEvaluationsOutput,
            )
            return eval_result.success
        except Exception as e:
            workflow.logger.warning(
                f"Evaluations failed for call_id={self._call_id}: {str(e)}"
            )
            return False

    async def _run_csat_evaluation(
        self, persist_result: FetchAndPersistCallResultOutput
    ) -> None:
        """Run CSAT evaluation if recording is available."""
        # Only run CSAT if there's a real conversation (both speakers present
        # or duration above the minimum threshold)
        has_meaningful_call = (
            persist_result.has_agent_message
            and persist_result.has_customer_message
        ) or (
            (self._call_duration_seconds or 0)
            >= MIN_DURATION_SECONDS_WITHOUT_TRANSCRIPT
            and persist_result.message_count > 0
        )

        if not has_meaningful_call:
            workflow.logger.info(
                f"Skipping CSAT: insufficient conversation data "
                f"for call_id={self._call_id}"
            )
            return

        try:
            csat_result = await workflow.execute_activity(
                "calculate_voice_csat_score",
                CalculateVoiceCSATInput(call_id=self._call_id),
                start_to_close_timeout=timedelta(minutes=10),
                heartbeat_timeout=timedelta(minutes=1),
                retry_policy=EVAL_RETRY_POLICY,
                task_queue=QUEUE_XL,
                result_type=CalculateVoiceCSATOutput,
            )
            if csat_result.success:
                workflow.logger.info(
                    f"CSAT score for call_id={self._call_id}: "
                    f"{csat_result.csat_score}"
                )
        except Exception as e:
            workflow.logger.warning(
                f"CSAT evaluation failed for call_id={self._call_id}: {str(e)}"
            )

    async def _calculate_conversation_metrics(
        self, prep_result: PrepareCallOutput
    ) -> None:
        """Calculate conversation metrics (talk ratio, interruptions, etc.)."""
        try:
            await workflow.execute_activity(
                "calculate_conversation_metrics",
                CalculateConversationMetricsInput(
                    call_id=self._call_id,
                    is_outbound=prep_result.is_outbound,
                    provider=prep_result.provider,
                    client_provider=(
                        prep_result.client_provider
                        if prep_result.client_provider != "vapi"
                        else None
                    ),
                ),
                start_to_close_timeout=timedelta(minutes=2),
                heartbeat_timeout=timedelta(minutes=1),
                retry_policy=DB_RETRY_POLICY,
                task_queue=QUEUE_L,
            )
        except Exception as e:
            workflow.logger.warning(
                f"Conversation metrics failed for call_id={self._call_id}: {str(e)}"
            )

    async def _run_tool_evaluation(
        self, prep_result: PrepareCallOutput
    ) -> None:
        """Run tool call evaluation if enabled for this call."""
        if not prep_result.enable_tool_evaluation:
            return

        try:
            tool_result = await workflow.execute_activity(
                "run_tool_call_evaluation",
                RunToolCallEvaluationInput(
                    call_execution_id=self._call_id,
                ),
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(minutes=1),
                retry_policy=EVAL_RETRY_POLICY,
                task_queue=QUEUE_XL,
                result_type=RunToolCallEvaluationOutput,
            )
            if tool_result.success:
                workflow.logger.info(
                    f"Tool evaluation completed for call_id={self._call_id}"
                )
        except Exception as e:
            workflow.logger.warning(
                f"Tool evaluation failed for call_id={self._call_id}: {str(e)}"
            )

    async def _complete(self) -> CallExecutionOutput:
        """Complete the workflow: signal parent, release slot, release phone."""
        self._status = "CLEANUP"

        # Signal parent workflow that call completed
        is_failed = self._call_status in ("failed", "cancelled")
        await workflow.execute_activity(
            "signal_call_completed",
            SignalCallCompleteInput(
                workflow_id=self._test_workflow_id,
                call_id=self._call_id,
                status=self._call_status or "ended",
                failed=is_failed,
            ),
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=SIGNAL_RETRY_POLICY,
            task_queue=QUEUE_S,
        )

        # Release rate limit slot
        await workflow.execute_activity(
            "release_call_slot",
            ReleaseSlotInput(call_id=self._call_id),
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=SIGNAL_RETRY_POLICY,
            task_queue=QUEUE_S,
        )

        # Release phone if acquired
        if self._phone_id:
            try:
                await workflow.execute_activity(
                    "release_phone_number",
                    ReleasePhoneInput(phone_id=self._phone_id),
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=DB_RETRY_POLICY,
                    task_queue=QUEUE_S,
                )
            except Exception as e:
                workflow.logger.warning(
                    f"Failed to release phone {self._phone_id}: {str(e)}"
                )

        final_status = (
            "COMPLETED"
            if self._call_status not in ("failed", "cancelled")
            else "FAILED"
        )
        self._status = final_status

        workflow.logger.info(
            f"CallExecutionWorkflow completed: call_id={self._call_id}, "
            f"status={final_status}"
        )

        return CallExecutionOutput(
            status=final_status,
            call_id=self._call_id,
            provider_call_id=self._provider_call_id,
            duration_seconds=self._call_duration_seconds,
            error=self._call_error,
        )

    async def _fail(self, error: str) -> CallExecutionOutput:
        """Fail the workflow with cleanup."""
        self._status = "FAILED"
        self._call_error = error

        workflow.logger.error(
            f"CallExecutionWorkflow failing: call_id={self._call_id}, error={error}"
        )

        # Always signal parent (even on failure) to prevent deadlock
        try:
            await workflow.execute_activity(
                "signal_call_completed",
                SignalCallCompleteInput(
                    workflow_id=self._test_workflow_id,
                    call_id=self._call_id,
                    status="failed",
                    failed=True,
                ),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=SIGNAL_RETRY_POLICY,
                task_queue=QUEUE_S,
            )
        except Exception:
            pass

        # Release slot if held
        try:
            await workflow.execute_activity(
                "release_call_slot",
                ReleaseSlotInput(call_id=self._call_id),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=SIGNAL_RETRY_POLICY,
                task_queue=QUEUE_S,
            )
        except Exception:
            pass

        # Release phone if acquired
        if self._phone_id:
            try:
                await workflow.execute_activity(
                    "release_phone_number",
                    ReleasePhoneInput(phone_id=self._phone_id),
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=DB_RETRY_POLICY,
                    task_queue=QUEUE_S,
                )
            except Exception:
                pass

        return CallExecutionOutput(
            status="FAILED",
            call_id=self._call_id,
            error=error,
        )

    async def _handle_cancellation(self) -> CallExecutionOutput:
        """Handle Temporal cancellation gracefully."""
        self._status = "CANCELLED"

        # Signal parent
        try:
            await workflow.execute_activity(
                "signal_call_completed",
                SignalCallCompleteInput(
                    workflow_id=self._test_workflow_id,
                    call_id=self._call_id,
                    status="cancelled",
                    failed=True,
                ),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=SIGNAL_RETRY_POLICY,
                task_queue=QUEUE_S,
            )
        except Exception:
            pass

        # Release slot
        try:
            await workflow.execute_activity(
                "release_call_slot",
                ReleaseSlotInput(call_id=self._call_id),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=SIGNAL_RETRY_POLICY,
                task_queue=QUEUE_S,
            )
        except Exception:
            pass

        # Release phone if acquired
        if self._phone_id:
            try:
                await workflow.execute_activity(
                    "release_phone_number",
                    ReleasePhoneInput(phone_id=self._phone_id),
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=DB_RETRY_POLICY,
                    task_queue=QUEUE_S,
                )
            except Exception:
                pass

        return CallExecutionOutput(
            status="CANCELLED",
            call_id=self._call_id,
            error="Cancelled by user",
        )
