"""
Module 5 — Parallel Execution Engine
======================================

Executes routed segments concurrently (where independent) or sequentially
(where dependent), respecting the dependency DAG from Module 2 and the
routing decisions from Module 4.

Paper reference  §3.3 "Asynchronous Orchestration":
    T_total ≈ max_i(T_gen(s_i)) + ε

Architecture:
  • Uses ``asyncio`` for cooperative concurrency.
  • Pluggable **backends**: swap between a mock backend (for testing)
    and a real Groq-API backend (for production) without touching
    the engine logic.
  • Per-segment **timing**: dispatch_time, start_time, first_token_time,
    end_time — enables latency analysis.
  • **Safety cancellation**: segments routed to ``safe_block`` are never
    dispatched; they receive a canned refusal.
  • **Escalation**: if a weak-tier call fails or times out, the engine
    optionally re-dispatches to the strong tier.

Execution policy:
  1. Walk the execution plan step by step.
  2. If a step's mode is ``"parallel"``, launch all its segments with
     ``asyncio.gather()`` — true concurrent I/O.
  3. If a step's mode is ``"sequential"``, execute one by one.
  4. Before dispatching, check the route tier:
       • ``safe_block``      → skip, emit refusal.
       • ``verify_required`` → dispatch to strong, flag for verification.
       • ``weak_model``      → dispatch to weak backend.
       • ``strong_model``    → dispatch to strong backend.
  5. Record wall-clock times at every transition.
  6. Collect all results, ordered by segment_id.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# Status constants
# ═══════════════════════════════════════════════════════════════════════

STATUS_PENDING   = "pending"
STATUS_SUCCESS   = "success"
STATUS_BLOCKED   = "blocked"
STATUS_FAILED    = "failed"
STATUS_ESCALATED = "escalated"
STATUS_TIMEOUT   = "timeout"


# ═══════════════════════════════════════════════════════════════════════
# Backend interface
# ═══════════════════════════════════════════════════════════════════════

class LLMBackend(ABC):
    """
    Abstract interface for an LLM inference backend.

    Subclass this to plug in Groq, OpenAI, vLLM, or any other provider.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        tier: str,
        segment_id: int,
        context: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Generate a response for a single segment.

        Parameters
        ----------
        prompt : str
            The segment text to respond to.
        tier : str
            ``"weak_model"`` or ``"strong_model"`` — determines which
            model endpoint to hit.
        segment_id : int
            Segment identifier (for logging / correlation).
        context : str, optional
            Upstream response text from a dependency (injected for
            dependent segments).
        timeout : float
            Maximum seconds to wait before raising TimeoutError.

        Returns
        -------
        dict with at least:
            response_text : str
            model_used    : str
            tokens_used   : int  (approximate)
        """
        ...


# ═══════════════════════════════════════════════════════════════════════
# Mock backend (for testing without API keys)
# ═══════════════════════════════════════════════════════════════════════

class MockLLMBackend(LLMBackend):
    """
    Simulated LLM backend that returns deterministic responses
    after a configurable delay.  Perfect for unit / integration tests.
    """

    def __init__(
        self,
        weak_delay: float = 0.05,
        strong_delay: float = 0.10,
        failure_ids: Optional[set] = None,
        timeout_ids: Optional[set] = None,
    ):
        """
        Parameters
        ----------
        weak_delay : float
            Simulated latency for weak-tier calls (seconds).
        strong_delay : float
            Simulated latency for strong-tier calls (seconds).
        failure_ids : set[int], optional
            Segment IDs that should raise RuntimeError (simulate failure).
        timeout_ids : set[int], optional
            Segment IDs that should raise asyncio.TimeoutError.
        """
        self.weak_delay = weak_delay
        self.strong_delay = strong_delay
        self.failure_ids = failure_ids or set()
        self.timeout_ids = timeout_ids or set()
        self.call_log: List[Dict[str, Any]] = []

    async def generate(
        self,
        prompt: str,
        tier: str,
        segment_id: int,
        context: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        delay = self.strong_delay if tier == "strong_model" else self.weak_delay

        # Record the call
        self.call_log.append({
            "segment_id": segment_id,
            "tier": tier,
            "prompt": prompt[:80],
            "context": context[:60] if context else None,
        })

        # Simulate timeout
        if segment_id in self.timeout_ids:
            await asyncio.sleep(timeout + 0.01)
            raise asyncio.TimeoutError(
                f"Segment {segment_id}: simulated timeout"
            )

        # Simulate failure
        if segment_id in self.failure_ids:
            await asyncio.sleep(delay * 0.5)
            raise RuntimeError(
                f"Segment {segment_id}: simulated backend failure"
            )

        # Normal response
        await asyncio.sleep(delay)

        model_name = (
            "llama-3-8b-instruct" if tier == "weak_model"
            else "llama-3-70b-instruct"
        )

        ctx_note = f" [with context: {context[:40]}...]" if context else ""
        response_text = (
            f"[Mock {model_name}] Response for segment {segment_id}: "
            f"\"{prompt[:60]}\"{ctx_note}"
        )

        return {
            "response_text": response_text,
            "model_used": model_name,
            "tokens_used": len(prompt.split()) * 3,  # rough estimate
        }


# ═══════════════════════════════════════════════════════════════════════
# Groq API backend (production)
# ═══════════════════════════════════════════════════════════════════════

class GroqLLMBackend(LLMBackend):
    """
    Real Groq API backend using the ``groq`` Python SDK.

    Requires:
      • ``pip install groq``
      • ``GROQ_API_KEY`` environment variable set.

    Model mapping:
      weak_model  → llama-3.1-8b-instant (or llama3-8b-8192)
      strong_model → llama-3.3-70b-versatile (or llama3-70b-8192)
    """

    WEAK_MODEL = "llama3-8b-8192"
    STRONG_MODEL = "llama3-70b-8192"

    def __init__(self, api_key: Optional[str] = None):
        import os
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import AsyncGroq
                self._client = AsyncGroq(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "groq package not installed. Run: pip install groq"
                )
        return self._client

    async def generate(
        self,
        prompt: str,
        tier: str,
        segment_id: int,
        context: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        client = self._get_client()
        model = (
            self.WEAK_MODEL if tier == "weak_model"
            else self.STRONG_MODEL
        )

        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": (
                    "You are a helpful assistant. The user's prior "
                    "sub-task produced this result which you may "
                    f"reference:\n\n{context}"
                ),
            })
        messages.append({"role": "user", "content": prompt})

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024,
                ),
                timeout=timeout,
            )

            text = response.choices[0].message.content or ""
            tokens = (
                response.usage.total_tokens
                if response.usage else len(text.split())
            )

            return {
                "response_text": text,
                "model_used": model,
                "tokens_used": tokens,
            }

        except asyncio.TimeoutError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Groq API error for segment {segment_id}: {e}"
            ) from e


# ═══════════════════════════════════════════════════════════════════════
# Segment execution result
# ═══════════════════════════════════════════════════════════════════════

def _empty_result(segment: Dict[str, Any]) -> Dict[str, Any]:
    """Create a blank execution result shell for a segment."""
    return {
        "segment_id": segment.get("segment_id", segment.get("id", -1)),
        "text": segment.get("text", ""),
        "route_tier": segment.get("route_tier", "unknown"),
        "intent_label": segment.get("intent_label", "unknown"),
        "complexity_band": segment.get("complexity_band", "unknown"),
        "response_text": "",
        "model_used": "",
        "tokens_used": 0,
        "status": "pending",         # pending | success | blocked | failed | escalated | timeout
        "error": None,
        "escalated": False,
        "timing": {
            "dispatch_time": None,
            "start_time": None,
            "first_token_time": None,
            "end_time": None,
            "latency_ms": None,
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════

class ParallelExecutionEngine:
    """
    Asynchronous execution engine that respects the dependency DAG.

    Usage::

        engine = ParallelExecutionEngine(backend=MockLLMBackend())

        # From the full pipeline:
        results = await engine.execute(
            execution_plan=graph["execution_plan"],
            routed_segments=routed_nodes,
        )

        # Or synchronously:
        results = engine.execute_sync(
            execution_plan=graph["execution_plan"],
            routed_segments=routed_nodes,
        )
    """

    BLOCKED_RESPONSE = (
        "⚠ This segment was blocked by the safety gate. "
        "The content has been flagged as potentially unsafe."
    )

    def __init__(
        self,
        backend: LLMBackend,
        enable_escalation: bool = True,
        escalation_timeout: float = 10.0,
        default_timeout: float = 30.0,
    ):
        """
        Parameters
        ----------
        backend : LLMBackend
            The inference backend (mock or real).
        enable_escalation : bool
            If True, weak-tier failures are retried on the strong tier.
        escalation_timeout : float
            Timeout (s) for escalation calls.
        default_timeout : float
            Default per-segment timeout (s).
        """
        self.backend = backend
        self.enable_escalation = enable_escalation
        self.escalation_timeout = escalation_timeout
        self.default_timeout = default_timeout

    # ══════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════

    async def execute(
        self,
        execution_plan: List[Dict[str, Any]],
        routed_segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute the full plan asynchronously.

        Parameters
        ----------
        execution_plan : list[dict]
            From ``graph["execution_plan"]`` (Module 2).
        routed_segments : list[dict]
            From ``router.route_all()`` (Module 4) — enriched with
            route_tier, route_confidence, etc.

        Returns
        -------
        dict with keys:
            results       : list[dict]  — per-segment execution results,
                            ordered by segment_id
            total_latency_ms : float
            parallel_latency_ms : float  — wall-clock time for the
                                           entire execution
            sequential_latency_ms : float — sum of all individual
                                            latencies (comparison baseline)
            speedup       : float  — sequential / parallel
            stats         : dict   — summary counts
        """
        wall_start = time.perf_counter()

        # Build a lookup: segment_id → routed segment
        seg_map: Dict[int, Dict[str, Any]] = {}
        for seg in routed_segments:
            sid = seg.get("segment_id", seg.get("id", -1))
            seg_map[sid] = seg

        # Results storage: segment_id → result dict
        results_map: Dict[int, Dict[str, Any]] = {}

        # Walk the execution plan step by step
        for step in execution_plan:
            step_segments = step["segments"]
            mode = step["mode"]

            if mode == "parallel" and len(step_segments) > 1:
                # Launch all segments in this step concurrently
                tasks = []
                for plan_seg in step_segments:
                    sid = plan_seg["segment_id"]
                    full_seg = seg_map.get(sid, plan_seg)
                    tasks.append(
                        self._execute_segment(full_seg, results_map)
                    )
                batch_results = await asyncio.gather(
                    *tasks, return_exceptions=False
                )
                for res in batch_results:
                    results_map[res["segment_id"]] = res
            else:
                # Sequential — execute one by one
                for plan_seg in step_segments:
                    sid = plan_seg["segment_id"]
                    full_seg = seg_map.get(sid, plan_seg)
                    res = await self._execute_segment(
                        full_seg, results_map
                    )
                    results_map[res["segment_id"]] = res

        wall_end = time.perf_counter()

        # Build ordered results list
        all_ids = sorted(results_map.keys())
        results_list = [results_map[sid] for sid in all_ids]

        # Compute aggregate timing
        parallel_ms = (wall_end - wall_start) * 1000
        sequential_ms = sum(
            r["timing"]["latency_ms"] or 0.0 for r in results_list
        )
        speedup = sequential_ms / parallel_ms if parallel_ms > 0 else 1.0

        stats = self._compute_stats(results_list)

        return {
            "results": results_list,
            "total_latency_ms": round(parallel_ms, 2),
            "parallel_latency_ms": round(parallel_ms, 2),
            "sequential_latency_ms": round(sequential_ms, 2),
            "speedup": round(speedup, 2),
            "stats": stats,
        }

    def execute_sync(
        self,
        execution_plan: List[Dict[str, Any]],
        routed_segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Synchronous wrapper around ``execute()``."""
        return asyncio.run(
            self.execute(execution_plan, routed_segments)
        )

    # ══════════════════════════════════════════════════════════════
    # Per-segment execution
    # ══════════════════════════════════════════════════════════════

    async def _execute_segment(
        self,
        segment: Dict[str, Any],
        completed: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute a single segment against the backend.

        Handles:
          • Safety blocks → canned refusal
          • Dependency context injection
          • Timeout / failure → optional escalation
          • Timing instrumentation
        """
        result = _empty_result(segment)
        sid = result["segment_id"]
        tier = segment.get("route_tier", "strong_model")

        # ── 1. Safety block shortcut ──────────────────────────────
        if tier == "safe_block":
            result["status"] = "blocked"
            result["response_text"] = self.BLOCKED_RESPONSE
            result["model_used"] = "safety_gate"
            result["timing"]["dispatch_time"] = time.perf_counter()
            result["timing"]["end_time"] = result["timing"]["dispatch_time"]
            result["timing"]["latency_ms"] = 0.0
            return result

        # ── 2. Build context from dependencies ───────────────────
        context = self._build_context(segment, completed)

        # ── 3. Dispatch ──────────────────────────────────────────
        dispatch_time = time.perf_counter()
        result["timing"]["dispatch_time"] = dispatch_time

        # Map verify_required to strong_model for actual API call
        actual_tier = "strong_model" if tier == "verify_required" else tier

        try:
            start_time = time.perf_counter()
            result["timing"]["start_time"] = start_time

            api_result = await asyncio.wait_for(
                self.backend.generate(
                    prompt=segment.get("text", ""),
                    tier=actual_tier,
                    segment_id=sid,
                    context=context,
                    timeout=self.default_timeout,
                ),
                timeout=self.default_timeout,
            )

            end_time = time.perf_counter()
            result["timing"]["end_time"] = end_time
            result["timing"]["first_token_time"] = start_time + (
                (end_time - start_time) * 0.1  # estimate: first token ~10%
            )
            result["timing"]["latency_ms"] = round(
                (end_time - dispatch_time) * 1000, 2
            )

            result["response_text"] = api_result["response_text"]
            result["model_used"] = api_result["model_used"]
            result["tokens_used"] = api_result.get("tokens_used", 0)
            result["status"] = "success"

        except (asyncio.TimeoutError, TimeoutError):
            result["status"] = "timeout"
            result["error"] = f"Segment {sid}: timed out after {self.default_timeout}s"
            result["timing"]["end_time"] = time.perf_counter()
            result["timing"]["latency_ms"] = round(
                (result["timing"]["end_time"] - dispatch_time) * 1000, 2
            )

            # Attempt escalation
            if self.enable_escalation and actual_tier == "weak_model":
                result = await self._escalate(segment, result, context)

        except Exception as e:
            result["status"] = "failed"
            result["error"] = f"Segment {sid}: {type(e).__name__}: {e}"
            result["timing"]["end_time"] = time.perf_counter()
            result["timing"]["latency_ms"] = round(
                (result["timing"]["end_time"] - dispatch_time) * 1000, 2
            )

            # Attempt escalation on weak-tier failures
            if self.enable_escalation and actual_tier == "weak_model":
                result = await self._escalate(segment, result, context)

        return result

    # ══════════════════════════════════════════════════════════════
    # Escalation (weak → strong on failure)
    # ══════════════════════════════════════════════════════════════

    async def _escalate(
        self,
        segment: Dict[str, Any],
        failed_result: Dict[str, Any],
        context: Optional[str],
    ) -> Dict[str, Any]:
        """
        Re-dispatch a failed weak-tier segment to the strong tier.
        """
        sid = segment.get("segment_id", segment.get("id", -1))
        result = dict(failed_result)
        result["escalated"] = True

        try:
            esc_start = time.perf_counter()

            api_result = await asyncio.wait_for(
                self.backend.generate(
                    prompt=segment.get("text", ""),
                    tier="strong_model",
                    segment_id=sid,
                    context=context,
                    timeout=self.escalation_timeout,
                ),
                timeout=self.escalation_timeout,
            )

            esc_end = time.perf_counter()

            result["response_text"] = api_result["response_text"]
            result["model_used"] = api_result["model_used"]
            result["tokens_used"] = api_result.get("tokens_used", 0)
            result["status"] = "escalated"
            result["error"] = None
            result["timing"]["end_time"] = esc_end
            result["timing"]["latency_ms"] = round(
                (esc_end - result["timing"]["dispatch_time"]) * 1000, 2
            )

        except Exception as esc_err:
            result["status"] = "failed"
            result["error"] = (
                f"Segment {sid}: original + escalation both failed: "
                f"{failed_result.get('error', '?')} → {esc_err}"
            )

        return result

    # ══════════════════════════════════════════════════════════════
    # Dependency context builder
    # ══════════════════════════════════════════════════════════════

    def _build_context(
        self,
        segment: Dict[str, Any],
        completed: Dict[int, Dict[str, Any]],
    ) -> Optional[str]:
        """
        Build context string from completed upstream dependencies.
        """
        depends_on = segment.get("depends_on", [])
        if not depends_on:
            return None

        context_parts = []
        for dep_id in depends_on:
            dep_result = completed.get(dep_id)
            if dep_result and dep_result.get("status") == "success":
                context_parts.append(
                    f"[Segment {dep_id}]: {dep_result['response_text']}"
                )
            elif dep_result and dep_result.get("status") == "escalated":
                context_parts.append(
                    f"[Segment {dep_id}]: {dep_result['response_text']}"
                )

        return "\n\n".join(context_parts) if context_parts else None

    # ══════════════════════════════════════════════════════════════
    # Statistics
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute execution summary statistics."""
        total = len(results)
        success = sum(1 for r in results if r["status"] == "success")
        blocked = sum(1 for r in results if r["status"] == "blocked")
        failed = sum(1 for r in results if r["status"] == "failed")
        escalated = sum(1 for r in results if r["status"] == "escalated")
        timed_out = sum(1 for r in results if r["status"] == "timeout")

        weak_calls = sum(
            1 for r in results
            if "8b" in r.get("model_used", "")
        )
        strong_calls = sum(
            1 for r in results
            if "70b" in r.get("model_used", "")
        )

        latencies = [
            r["timing"]["latency_ms"] for r in results
            if r["timing"]["latency_ms"] is not None and r["status"] in ("success", "escalated")
        ]

        return {
            "total_segments": total,
            "success": success,
            "blocked": blocked,
            "failed": failed,
            "escalated": escalated,
            "timed_out": timed_out,
            "weak_calls": weak_calls,
            "strong_calls": strong_calls,
            "avg_latency_ms": round(
                sum(latencies) / len(latencies), 2
            ) if latencies else 0.0,
            "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
            "min_latency_ms": round(min(latencies), 2) if latencies else 0.0,
        }

    # ══════════════════════════════════════════════════════════════
    # Pretty-print
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def print_results(
        output: Dict[str, Any],
        show_responses: bool = True,
    ) -> None:
        """Print a human-readable execution report."""
        print("=" * 70)
        print("EXECUTION RESULTS")
        print("=" * 70)

        results = output["results"]
        for res in results:
            sid = res["segment_id"]
            status = res["status"]
            tier = res["route_tier"]
            model = res["model_used"]
            latency = res["timing"]["latency_ms"]
            esc = " [ESCALATED]" if res.get("escalated") else ""

            status_icon = {
                "success": "✅",
                "blocked": "🚫",
                "failed": "❌",
                "escalated": "🔄",
                "timeout": "⏰",
                "pending": "⏳",
            }.get(status, "❓")

            print(f"\n  [{sid}] {status_icon} {status}{esc}")
            print(f"       text    = \"{res['text'][:65]}{'…' if len(res['text']) > 65 else ''}\"")
            print(f"       tier    = {tier} → model={model}")
            print(f"       latency = {latency} ms")

            if res.get("error"):
                print(f"       ⚠ error = {res['error'][:80]}")

            if show_responses and res["response_text"]:
                resp_preview = res["response_text"][:100]
                print(f"       response= \"{resp_preview}{'…' if len(res['response_text']) > 100 else ''}\"")

        # Summary
        stats = output["stats"]
        print(f"\n  ── Timing ──")
        print(f"     Parallel (wall-clock): {output['parallel_latency_ms']:.1f} ms")
        print(f"     Sequential (sum):      {output['sequential_latency_ms']:.1f} ms")
        print(f"     Speedup:               {output['speedup']:.2f}×")

        print(f"\n  ── Stats ──")
        print(f"     Total: {stats['total_segments']}  "
              f"Success: {stats['success']}  "
              f"Blocked: {stats['blocked']}  "
              f"Failed: {stats['failed']}  "
              f"Escalated: {stats['escalated']}  "
              f"Timeout: {stats['timed_out']}")
        print(f"     Weak calls: {stats['weak_calls']}  "
              f"Strong calls: {stats['strong_calls']}")

        if stats["avg_latency_ms"] > 0:
            print(f"     Latency: avg={stats['avg_latency_ms']:.1f} ms  "
                  f"min={stats['min_latency_ms']:.1f} ms  "
                  f"max={stats['max_latency_ms']:.1f} ms")

        print("\n" + "=" * 70)
