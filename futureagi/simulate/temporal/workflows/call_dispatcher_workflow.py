"""
CallDispatcherWorkflow - Singleton rate-limiting dispatcher for concurrent calls.

Manages call slot allocation with three-tier rate limiting:
1. App-level: maximum concurrent calls across all organizations
2. Org-level: maximum concurrent calls per organization
3. Agent-level: maximum concurrent calls per agent definition (LiveKit)

Architecture:
- Singleton workflow (one per Temporal namespace)
- FIFO queuing with per-org and per-agent limits
- Signal-driven: receives request_slot, release_slot, update_limits signals
- Batched slot grants for efficiency
- Stale slot reaper for dead child workflows
- Continue-as-new for long-running dispatcher lifecycle
- Custom shutdown handler for clean service shutdown

Design:
- Simple, predictable FIFO ordering — no priority inversion
- Per-org isolation: one org's traffic can't starve others
- Agent concurrency: prevents overwhelming LiveKit agent workers
- Signal batching: grants are batched and delivered together to reduce activity overhead
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from temporalio import workflow

from simulate.temporal.constants import (
    DEFAULT_APP_LIMIT,
    DEFAULT_ORG_LIMIT,
    DISPATCHER_CONTINUE_AS_NEW_THRESHOLD,
    QUEUE_S,
    STALE_SLOT_REAP_INTERVAL_SECONDS,
    STALE_SLOT_THRESHOLD_SECONDS,
)
from simulate.temporal.retry_policies import SIGNAL_RETRY_POLICY
from simulate.temporal.signals import (
    SIGNAL_RELEASE_SLOT,
    SIGNAL_REQUEST_SLOT,
    SIGNAL_UPDATE_LIMITS,
)
from simulate.temporal.types.activities import (
    SignalSlotBatchInput,
)
from simulate.temporal.types.dispatcher import (
    ActiveCall,
    DispatcherState,
    DispatcherStatus,
)


# Grant batch size: number of slots to grant per iteration.
# Larger batches reduce activity overhead but may cause bursty call initiations.
GRANT_BATCH_SIZE = 10

# Main loop sleep interval (seconds) between processing iterations.
# Lower values reduce latency but increase Temporal event count.
MAIN_LOOP_SLEEP_SECONDS = 0.5


@workflow.defn
class CallDispatcherWorkflow:
    """
    Singleton rate-limiting dispatcher for concurrent call execution.

    Enforces three-tier limits:
    - App-wide concurrency (DEFAULT_APP_LIMIT, default 100)
    - Per-org concurrency (DEFAULT_ORG_LIMIT, default 25)
    - Per-agent concurrency (dynamic, set per agent definition)

    State persists across continue-as-new via DispatcherState.
    """

    def __init__(self):
        self._pending_queue: list[dict] = []
        self._active_calls: dict[str, ActiveCall] = {}
        self._org_call_counts: dict[str, int] = {}
        self._agent_call_counts: dict[str, int] = {}
        self._agent_limits: dict[str, int] = {}

        # Rate limits
        self._app_limit: int = DEFAULT_APP_LIMIT
        self._org_limit: int = DEFAULT_ORG_LIMIT

        # Lifetime counters
        self._total_granted: int = 0
        self._total_released: int = 0

        # Stale slot reaper state
        self._last_reap_time: float = 0.0

        # Shutdown coordination
        self._shutdown_requested: bool = False

        # Pending grants that haven't been signaled yet
        # Preserved across continue-as-new to prevent slot leaks
        self._pending_grants: list[dict] = []

    @workflow.run
    async def run(self, state: Optional[DispatcherState] = None) -> None:
        """
        Main dispatcher loop.

        Runs indefinitely as a singleton. Processes slot requests,
        grants slots when capacity is available, and runs periodic
        stale slot reaping.

        Args:
            state: DispatcherState from continue-as-new, or None for fresh start.
        """
        if state:
            self._restore_state(state)
            workflow.logger.info(
                f"Dispatcher restored: pending={len(self._pending_queue)}, "
                f"active={len(self._active_calls)}, granted={self._total_granted}"
            )

        workflow.logger.info(
            f"CallDispatcherWorkflow started: app_limit={self._app_limit}, "
            f"org_limit={self._org_limit}"
        )

        try:
            await self._main_loop()
        except asyncio.CancelledError:
            workflow.logger.info("CallDispatcherWorkflow received CancelledError")
            if not self._shutdown_requested:
                # Unexpected cancellation — checkpoint state
                await self._checkpoint()
            # If shutdown was requested, let the cancellation propagate
            # so Temporal can start a new instance with the saved state

    async def _main_loop(self) -> None:
        """Main processing loop — runs until shutdown or continue-as-new."""
        event_count = 0

        while not self._shutdown_requested:
            # Check for continue-as-new
            if event_count >= DISPATCHER_CONTINUE_AS_NEW_THRESHOLD:
                await self._checkpoint()
                return  # Unreachable; checkpoint calls continue_as_new

            # Process pending slot grants (from previous iteration's continue-as-new)
            if self._pending_grants:
                await self._signal_pending_grants()
                event_count += 1

            # Grant slots from pending queue
            if self._pending_queue:
                granted = await self._process_slot_grants()
                event_count += granted

            # Run stale slot reaper periodically
            now = workflow.time()
            if now - self._last_reap_time >= STALE_SLOT_REAP_INTERVAL_SECONDS:
                await self._reap_stale_slots()
                self._last_reap_time = now
                event_count += 1

            # Sleep between iterations to prevent tight loops
            await workflow.sleep(MAIN_LOOP_SLEEP_SECONDS)
            event_count += 1

    # ========================================
    # SIGNAL HANDLERS
    # ========================================

    @workflow.signal
    async def request_slot(self, signal: dict) -> None:
        """
        Signal: Request a call slot.

        Expected dict keys:
            workflow_id: str    — CallExecutionWorkflow ID to signal on grant
            call_id: str        — Call identifier
            org_id: str         — Organization ID for rate limiting
            priority: int       — Priority (0=normal, ignored in FIFO mode)
            agent_definition_id: str (optional) — For per-agent concurrency
            agent_concurrency_limit: int (optional) — Max concurrent for this agent
        """
        call_id = signal.get("call_id", "unknown")
        org_id = signal.get("org_id", "unknown")

        workflow.logger.info(
            f"Slot requested: call_id={call_id}, org_id={org_id}, "
            f"queue_size={len(self._pending_queue)}"
        )

        # Register agent limits if provided
        agent_def_id = signal.get("agent_definition_id")
        agent_limit = signal.get("agent_concurrency_limit")
        if agent_def_id and agent_limit:
            self._agent_limits[agent_def_id] = agent_limit

        # Add to pending queue
        self._pending_queue.append({
            "workflow_id": signal["workflow_id"],
            "call_id": signal["call_id"],
            "org_id": signal["org_id"],
            "priority": signal.get("priority", 0),
            "agent_definition_id": signal.get("agent_definition_id"),
        })

    @workflow.signal
    async def release_slot(self, call_id: str) -> None:
        """
        Signal: Release a call slot.

        Frees up capacity so the next pending request can be granted.
        Called by CallExecutionWorkflow when a call completes or fails.
        """
        workflow.logger.info(
            f"Slot release signal: call_id={call_id}"
        )

        if call_id in self._active_calls:
            active = self._active_calls.pop(call_id)

            # Decrement org count
            org_id = active.org_id
            if org_id in self._org_call_counts:
                self._org_call_counts[org_id] = max(
                    0, self._org_call_counts[org_id] - 1
                )

            # Decrement agent count
            if active.agent_definition_id:
                agent_id = active.agent_definition_id
                if agent_id in self._agent_call_counts:
                    self._agent_call_counts[agent_id] = max(
                        0, self._agent_call_counts[agent_id] - 1
                    )

            self._total_released += 1
        else:
            # Slot may have been already released or call was never tracked
            workflow.logger.warning(
                f"Release for unknown call_id={call_id} (already released?)"
            )

    @workflow.signal
    async def update_limits(self, limits: dict) -> None:
        """
        Signal: Update rate limits dynamically.

        Expected dict keys:
            app_limit: int (optional) — New app-wide concurrency limit
            org_limit: int (optional) — New per-org concurrency limit
            agent_limits: dict[str, int] (optional) — New per-agent limits
        """
        if "app_limit" in limits:
            self._app_limit = limits["app_limit"]
            workflow.logger.info(f"Updated app_limit to {self._app_limit}")
        if "org_limit" in limits:
            self._org_limit = limits["org_limit"]
            workflow.logger.info(f"Updated org_limit to {self._org_limit}")
        if "agent_limits" in limits:
            self._agent_limits.update(limits["agent_limits"])
            workflow.logger.info(
                f"Updated agent_limits for {len(limits['agent_limits'])} agents"
            )

    @workflow.signal
    async def shutdown(self) -> None:
        """
        Signal: Request graceful shutdown.

        Sets the shutdown flag so the main loop exits cleanly.
        State is checkpointed via continue-as-new before exit.
        """
        self._shutdown_requested = True
        workflow.logger.info(
            f"Shutdown requested: pending={len(self._pending_queue)}, "
            f"active={len(self._active_calls)}"
        )

    @workflow.signal
    async def request_slots_batch(self, requests: list[dict]) -> None:
        """
        Signal: Batch request for multiple call slots.

        More efficient than individual request_slot signals for
        bulk operations. Each item in the list has the same format
        as individual request_slot signals.
        """
        workflow.logger.info(
            f"Batch slot request: count={len(requests)}"
        )

        for req in requests:
            # Register agent limits
            agent_def_id = req.get("agent_definition_id")
            agent_limit = req.get("agent_concurrency_limit")
            if agent_def_id and agent_limit:
                self._agent_limits[agent_def_id] = agent_limit

            self._pending_queue.append({
                "workflow_id": req["workflow_id"],
                "call_id": req["call_id"],
                "org_id": req["org_id"],
                "priority": req.get("priority", 0),
                "agent_definition_id": req.get("agent_definition_id"),
            })

    # ========================================
    # QUERIES
    # ========================================

    @workflow.query
    def get_status(self) -> DispatcherStatus:
        """Query current dispatcher status."""
        return DispatcherStatus(
            pending_count=len(self._pending_queue),
            active_count=len(self._active_calls),
            org_counts={
                org_id: count
                for org_id, count in self._org_call_counts.items()
                if count > 0
            },
            app_limit=self._app_limit,
            org_limit=self._org_limit,
            total_granted=self._total_granted,
            total_released=self._total_released,
            orgs_with_pending=len(
                set(req["org_id"] for req in self._pending_queue)
            ),
        )

    # ========================================
    # INTERNAL METHODS
    # ========================================

    async def _process_slot_grants(self) -> int:
        """
        Process pending queue and grant slots up to capacity.

        Returns the number of slots granted in this iteration.
        """
        granted_this_iteration = 0
        grants: list[dict] = []

        i = 0
        while i < len(self._pending_queue) and granted_this_iteration < GRANT_BATCH_SIZE:
            # Check app-level limit
            if len(self._active_calls) + len(grants) >= self._app_limit:
                break

            req = self._pending_queue[i]
            org_id = req["org_id"]
            agent_def_id = req.get("agent_definition_id")

            # Check org-level limit
            current_org_count = self._org_call_counts.get(org_id, 0)
            org_pending_grants = sum(
                1 for g in grants if g["org_id"] == org_id
            )
            if current_org_count + org_pending_grants >= self._org_limit:
                i += 1
                continue

            # Check agent-level limit
            if agent_def_id:
                agent_limit = self._agent_limits.get(agent_def_id, 0)
                if agent_limit > 0:
                    current_agent_count = self._agent_call_counts.get(
                        agent_def_id, 0
                    )
                    agent_pending_grants = sum(
                        1 for g in grants
                        if g.get("agent_definition_id") == agent_def_id
                    )
                    if current_agent_count + agent_pending_grants >= agent_limit:
                        i += 1
                        continue

            # Grant the slot
            call_id = req["call_id"]
            now_iso = datetime.now(timezone.utc).isoformat()

            # Track active call
            self._active_calls[call_id] = ActiveCall(
                call_id=call_id,
                org_id=org_id,
                granted_at=now_iso,
                agent_definition_id=agent_def_id,
            )

            # Update counters
            self._org_call_counts[org_id] = (
                self._org_call_counts.get(org_id, 0) + 1
            )
            if agent_def_id:
                self._agent_call_counts[agent_def_id] = (
                    self._agent_call_counts.get(agent_def_id, 0) + 1
                )
            self._total_granted += 1

            # Add to batched grants
            grants.append({
                "workflow_id": req["workflow_id"],
                "call_id": call_id,
            })

            # Remove from pending
            self._pending_queue.pop(i)
            granted_this_iteration += 1
            # Don't increment i since we removed an element

        # Signal all granted workflows as a batch
        if grants:
            await self._signal_grants_batch(grants)

        return granted_this_iteration

    async def _signal_grants_batch(self, grants: list[dict]) -> None:
        """
        Send SLOT_GRANTED signals to all granted workflows as a batch.

        If the batch activity fails, grants are preserved in _pending_grants
        and retried on the next iteration. This prevents slot leaks where
        slots are tracked but workflows never receive the grant signal.
        """
        # Save grants as pending before attempting to signal.
        # On failure, these will be retried next iteration.
        self._pending_grants.extend(grants)

        try:
            await workflow.execute_activity(
                "signal_slots_granted_batch",
                SignalSlotBatchInput(grants=grants),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=SIGNAL_RETRY_POLICY,
                task_queue=QUEUE_S,
            )

            # Success — clear pending grants
            self._pending_grants.clear()

            workflow.logger.info(
                f"Granted {len(grants)} slots: "
                f"active={len(self._active_calls)}, "
                f"pending={len(self._pending_queue)}"
            )
        except Exception as e:
            workflow.logger.warning(
                f"Failed to signal grants batch ({len(grants)} slots): {e}. "
                f"Will retry next iteration."
            )
            # pending_grants preserved for retry

    async def _signal_pending_grants(self) -> None:
        """Retry signaling previously failed batch grants."""
        grants = self._pending_grants.copy()
        workflow.logger.info(
            f"Retrying {len(grants)} pending slot grants"
        )
        await self._signal_grants_batch(grants)

    async def _reap_stale_slots(self) -> None:
        """
        Release slots that have been active longer than the stale threshold.

        This prevents slot leaks when CallExecutionWorkflows die without
        properly releasing their slots (worker crash, network partition, etc.).
        """
        now = workflow.time()
        stale_threshold = STALE_SLOT_THRESHOLD_SECONDS
        stale_call_ids: list[str] = []

        for call_id, active in list(self._active_calls.items()):
            # Parse granted_at time and compute age
            try:
                granted_dt = datetime.fromisoformat(active.granted_at)
                age_seconds = now - granted_dt.timestamp()
            except (ValueError, TypeError):
                # Can't parse timestamp — assume stale
                workflow.logger.warning(
                    f"Unparseable granted_at for call_id={call_id}: "
                    f"{active.granted_at}"
                )
                stale_call_ids.append(call_id)
                continue

            if age_seconds > stale_threshold:
                stale_call_ids.append(call_id)
                workflow.logger.warning(
                    f"Stale slot detected: call_id={call_id}, "
                    f"age={age_seconds:.0f}s, threshold={stale_threshold}s"
                )

        if stale_call_ids:
            workflow.logger.warning(
                f"Reaping {len(stale_call_ids)} stale slots"
            )
            for call_id in stale_call_ids:
                # Use the same cleanup logic as release_slot
                if call_id in self._active_calls:
                    active = self._active_calls.pop(call_id)
                    org_id = active.org_id
                    if org_id in self._org_call_counts:
                        self._org_call_counts[org_id] = max(
                            0, self._org_call_counts[org_id] - 1
                        )
                    if active.agent_definition_id:
                        agent_id = active.agent_definition_id
                        if agent_id in self._agent_call_counts:
                            self._agent_call_counts[agent_id] = max(
                                0, self._agent_call_counts[agent_id] - 1
                            )

    def _restore_state(self, state: DispatcherState) -> None:
        """Restore state from continue-as-new checkpoint."""
        self._pending_queue = [
            {
                "workflow_id": req.workflow_id,
                "call_id": req.call_id,
                "org_id": req.org_id,
                "priority": 0,
                "agent_definition_id": req.agent_definition_id,
            }
            for req in state.pending_queue
        ]
        self._active_calls = state.active_calls.copy()
        self._org_call_counts = state.org_call_counts.copy()
        self._agent_call_counts = state.agent_call_counts.copy()
        self._agent_limits = state.agent_limits.copy()
        self._app_limit = state.app_limit
        self._org_limit = state.org_limit
        self._total_granted = state.total_granted
        self._total_released = state.total_released
        self._pending_grants = state.pending_grants.copy()

    async def _checkpoint(self) -> None:
        """
        Checkpoint and continue-as-new to prevent unbounded history growth.

        The dispatcher is a singleton — it never "completes". Continue-as-new
        keeps history bounded while preserving all state.
        """
        workflow.logger.info(
            f"Dispatcher continue-as-new: pending={len(self._pending_queue)}, "
            f"active={len(self._active_calls)}, granted={self._total_granted}"
        )

        state = self._build_state()
        workflow.continue_as_new(state)

    def _build_state(self) -> DispatcherState:
        """Build DispatcherState for continue-as-new."""
        from simulate.temporal.types.dispatcher import SlotRequest

        return DispatcherState(
            pending_queue=[
                SlotRequest(
                    call_id=req["call_id"],
                    org_id=req["org_id"],
                    workflow_id=req["workflow_id"],
                    agent_definition_id=req.get("agent_definition_id"),
                )
                for req in self._pending_queue
            ],
            pending_count=len(self._pending_queue),
            active_calls=self._active_calls.copy(),
            org_call_counts=self._org_call_counts.copy(),
            agent_call_counts=self._agent_call_counts.copy(),
            agent_limits=self._agent_limits.copy(),
            app_limit=self._app_limit,
            org_limit=self._org_limit,
            total_granted=self._total_granted,
            total_released=self._total_released,
            pending_grants=self._pending_grants.copy(),
        )
