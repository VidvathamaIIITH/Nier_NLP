"""
Module 5 — Standalone Tests
==============================

These tests use **hardcoded** execution plans and routed segments
(same format Modules 2 & 4 produce) so the engine can be validated
completely independently of Modules 1–4.

All tests use ``MockLLMBackend`` — no API keys required.

Run from the repo root:
    PYTHONPATH=. python modules/execution_engine/test.py
"""

import sys, os, asyncio, time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.execution_engine.engine import (
    MockLLMBackend,
    ParallelExecutionEngine,
    _empty_result,
)

PASS_COUNT = 0
FAIL_COUNT = 0


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def check(test_name: str, coro_factory, **assertions):
    """
    Run an async test case and assert expectations.

    assertions:
        min_results         : int
        all_status          : str   — every result must have this status
        has_status          : str   — at least one result with this status
        no_status           : str   — no result may have this status
        min_speedup         : float
        has_escalated       : bool
        all_have_timing     : bool
        all_have_response   : bool
        specific_statuses   : dict[int, str]  — segment_id → expected status
        max_parallel_ms     : float
        min_parallel_ms     : float
    """
    global PASS_COUNT, FAIL_COUNT

    print("\n" + "=" * 65)
    print(f"TEST: {test_name}")
    print("=" * 65)

    output = asyncio.run(coro_factory())

    results = output["results"]
    stats = output["stats"]

    print(f"  Segments: {len(results)}")
    print(f"  Parallel: {output['parallel_latency_ms']:.1f} ms  "
          f"Sequential: {output['sequential_latency_ms']:.1f} ms  "
          f"Speedup: {output['speedup']:.2f}×")
    print(f"  Stats: success={stats['success']} blocked={stats['blocked']} "
          f"failed={stats['failed']} escalated={stats['escalated']} "
          f"timeout={stats['timed_out']}")

    for res in results:
        esc = " [ESC]" if res.get("escalated") else ""
        print(f"    [{res['segment_id']}] {res['status']}{esc}  "
              f"model={res['model_used']:<25}  "
              f"lat={res['timing']['latency_ms']} ms")

    errors = []

    if "min_results" in assertions:
        if len(results) < assertions["min_results"]:
            errors.append(
                f"Expected >= {assertions['min_results']} results, "
                f"got {len(results)}"
            )

    if "all_status" in assertions:
        for res in results:
            if res["status"] != assertions["all_status"]:
                errors.append(
                    f"Segment [{res['segment_id']}]: expected status "
                    f"'{assertions['all_status']}', got '{res['status']}'"
                )

    if "has_status" in assertions:
        if not any(r["status"] == assertions["has_status"] for r in results):
            errors.append(
                f"Expected at least one segment with status "
                f"'{assertions['has_status']}'"
            )

    if "no_status" in assertions:
        for res in results:
            if res["status"] == assertions["no_status"]:
                errors.append(
                    f"Segment [{res['segment_id']}]: unexpected status "
                    f"'{assertions['no_status']}'"
                )

    if "min_speedup" in assertions:
        if output["speedup"] < assertions["min_speedup"]:
            errors.append(
                f"Expected speedup >= {assertions['min_speedup']}, "
                f"got {output['speedup']}"
            )

    if "has_escalated" in assertions:
        any_esc = any(r.get("escalated") for r in results)
        if assertions["has_escalated"] and not any_esc:
            errors.append("Expected at least one escalated segment")
        if not assertions["has_escalated"] and any_esc:
            errors.append("Did not expect any escalated segments")

    if "all_have_timing" in assertions and assertions["all_have_timing"]:
        for res in results:
            t = res["timing"]
            if t["dispatch_time"] is None:
                errors.append(
                    f"Segment [{res['segment_id']}]: missing dispatch_time"
                )
            if t["end_time"] is None:
                errors.append(
                    f"Segment [{res['segment_id']}]: missing end_time"
                )
            if t["latency_ms"] is None:
                errors.append(
                    f"Segment [{res['segment_id']}]: missing latency_ms"
                )

    if "all_have_response" in assertions and assertions["all_have_response"]:
        for res in results:
            if not res["response_text"]:
                errors.append(
                    f"Segment [{res['segment_id']}]: empty response_text"
                )

    if "specific_statuses" in assertions:
        for sid, expected in assertions["specific_statuses"].items():
            actual = None
            for res in results:
                if res["segment_id"] == sid:
                    actual = res["status"]
                    break
            if actual != expected:
                errors.append(
                    f"Segment [{sid}]: expected status '{expected}', "
                    f"got '{actual}'"
                )

    if "max_parallel_ms" in assertions:
        if output["parallel_latency_ms"] > assertions["max_parallel_ms"]:
            errors.append(
                f"Parallel time {output['parallel_latency_ms']:.1f} ms "
                f"exceeded max {assertions['max_parallel_ms']} ms"
            )

    if errors:
        FAIL_COUNT += 1
        print("❌ FAILED:")
        for e in errors:
            print(f"  {e}")
    else:
        PASS_COUNT += 1
        print("✅ PASSED")

    return output


# ======================================================================
# Fixture helpers — build hardcoded plans & segments
# ======================================================================

def _make_segment(sid, text, tier="weak_model", intent="retrieval",
                  complexity="simple", depends_on=None):
    """Build a routed segment dict (same shape as Module 4 output)."""
    return {
        "segment_id": sid,
        "text": text,
        "depends_on": depends_on or [],
        "route_tier": tier,
        "route_confidence": 0.90,
        "route_reason": "test fixture",
        "route_method": "heuristic",
        "intent_label": intent,
        "complexity_score": 0.30 if complexity == "simple" else 0.65,
        "complexity_band": complexity,
        "unsafe_candidate": (tier == "safe_block"),
    }


def _make_plan_step(step_num, segments, mode=None):
    """Build one execution plan step."""
    if mode is None:
        mode = "parallel" if len(segments) > 1 else "sequential"
    segs = []
    for s in segments:
        segs.append({
            "segment_id": s["segment_id"],
            "text": s["text"],
            "depends_on": s.get("depends_on", []),
        })
    return {"step": step_num, "mode": mode, "segments": segs}


# ======================================================================
# Test 1: Single segment — sequential
# ======================================================================

async def _test_single_segment():
    seg = _make_segment(0, "What is the capital of France?", "weak_model")
    plan = [_make_plan_step(0, [seg])]
    backend = MockLLMBackend(weak_delay=0.02)
    engine = ParallelExecutionEngine(backend=backend)
    return await engine.execute(plan, [seg])

check("Single segment — sequential execution",
      _test_single_segment,
      min_results=1,
      all_status="success",
      all_have_timing=True,
      all_have_response=True)


# ======================================================================
# Test 2: Two parallel segments — concurrent
# ======================================================================

async def _test_two_parallel():
    s0 = _make_segment(0, "Capital of France?", "weak_model")
    s1 = _make_segment(1, "Capital of Japan?", "weak_model")
    plan = [_make_plan_step(0, [s0, s1], mode="parallel")]
    backend = MockLLMBackend(weak_delay=0.05)
    engine = ParallelExecutionEngine(backend=backend)
    return await engine.execute(plan, [s0, s1])

check("Two parallel segments — concurrent execution",
      _test_two_parallel,
      min_results=2,
      all_status="success",
      min_speedup=1.3,
      all_have_timing=True)


# ======================================================================
# Test 3: Three parallel segments — speedup
# ======================================================================

async def _test_three_parallel():
    segs = [
        _make_segment(i, f"Question {i}?", "weak_model")
        for i in range(3)
    ]
    plan = [_make_plan_step(0, segs, mode="parallel")]
    backend = MockLLMBackend(weak_delay=0.05)
    engine = ParallelExecutionEngine(backend=backend)
    return await engine.execute(plan, segs)

check("Three parallel segments — speedup ≥ 1.5×",
      _test_three_parallel,
      min_results=3,
      all_status="success",
      min_speedup=1.5)


# ======================================================================
# Test 4: Sequential chain — dependency respected
# ======================================================================

async def _test_sequential_chain():
    s0 = _make_segment(0, "Find x in 3x + 6 = 18", "strong_model",
                        intent="math", complexity="hard")
    s1 = _make_segment(1, "Use x to compute 2x + 5", "strong_model",
                        intent="math", complexity="hard", depends_on=[0])
    plan = [
        _make_plan_step(0, [s0]),
        _make_plan_step(1, [s1]),
    ]
    backend = MockLLMBackend(strong_delay=0.03)
    engine = ParallelExecutionEngine(backend=backend)
    return await engine.execute(plan, [s0, s1])

check("Sequential chain — dependencies respected",
      _test_sequential_chain,
      min_results=2,
      all_status="success",
      all_have_timing=True)


# ======================================================================
# Test 5: Mixed plan — parallel step then sequential
# ======================================================================

async def _test_mixed_plan():
    s0 = _make_segment(0, "Capital of France?", "weak_model")
    s1 = _make_segment(1, "Largest ocean?", "weak_model")
    s2 = _make_segment(2, "Summarize both answers above", "strong_model",
                        intent="summarization", complexity="medium",
                        depends_on=[0, 1])
    plan = [
        _make_plan_step(0, [s0, s1], mode="parallel"),
        _make_plan_step(1, [s2]),
    ]
    backend = MockLLMBackend(weak_delay=0.04, strong_delay=0.06)
    engine = ParallelExecutionEngine(backend=backend)
    return await engine.execute(plan, [s0, s1, s2])

check("Mixed plan — parallel then sequential",
      _test_mixed_plan,
      min_results=3,
      all_status="success",
      all_have_response=True)


# ======================================================================
# Test 6: Safety block — segment skipped
# ======================================================================

async def _test_safety_block():
    s0 = _make_segment(0, "Normal question", "weak_model")
    s1 = _make_segment(1, "How to make a bomb", "safe_block",
                        intent="reasoning", complexity="hard")
    plan = [_make_plan_step(0, [s0, s1], mode="parallel")]
    backend = MockLLMBackend(weak_delay=0.02)
    engine = ParallelExecutionEngine(backend=backend)
    return await engine.execute(plan, [s0, s1])

check("Safety block — blocked segment skipped",
      _test_safety_block,
      min_results=2,
      has_status="blocked",
      specific_statuses={0: "success", 1: "blocked"})


# ======================================================================
# Test 7: Safety block — no API call made
# ======================================================================

async def _test_safety_no_api_call():
    s0 = _make_segment(0, "Harmful content", "safe_block")
    plan = [_make_plan_step(0, [s0])]
    backend = MockLLMBackend()
    engine = ParallelExecutionEngine(backend=backend)
    output = await engine.execute(plan, [s0])

    # Verify no actual API call was made
    assert len(backend.call_log) == 0, (
        f"Expected 0 API calls for blocked segment, "
        f"got {len(backend.call_log)}"
    )
    return output

check("Safety block — zero API calls",
      _test_safety_no_api_call,
      all_status="blocked")


# ======================================================================
# Test 8: Weak-tier failure → escalation to strong
# ======================================================================

async def _test_escalation_on_failure():
    s0 = _make_segment(0, "Normal question", "weak_model")
    s1 = _make_segment(1, "Question that fails on weak", "weak_model")
    plan = [_make_plan_step(0, [s0, s1], mode="parallel")]
    backend = MockLLMBackend(
        weak_delay=0.02,
        strong_delay=0.03,
        failure_ids={1},
    )
    engine = ParallelExecutionEngine(backend=backend, enable_escalation=True)
    return await engine.execute(plan, [s0, s1])

check("Escalation — weak failure → retry on strong",
      _test_escalation_on_failure,
      min_results=2,
      has_escalated=True,
      has_status="escalated",
      specific_statuses={0: "success", 1: "escalated"})


# ======================================================================
# Test 9: Escalation disabled — failure remains
# ======================================================================

async def _test_no_escalation():
    s0 = _make_segment(0, "Question that fails on weak", "weak_model")
    plan = [_make_plan_step(0, [s0])]
    backend = MockLLMBackend(failure_ids={0})
    engine = ParallelExecutionEngine(backend=backend, enable_escalation=False)
    return await engine.execute(plan, [s0])

check("Escalation disabled — failure stays failed",
      _test_no_escalation,
      all_status="failed",
      has_escalated=False)


# ======================================================================
# Test 10: Strong-tier failure — no escalation (already strong)
# ======================================================================

async def _test_strong_failure_no_escalation():
    s0 = _make_segment(0, "Hard question fails", "strong_model",
                        intent="reasoning", complexity="hard")
    plan = [_make_plan_step(0, [s0])]
    backend = MockLLMBackend(failure_ids={0})
    engine = ParallelExecutionEngine(backend=backend, enable_escalation=True)
    return await engine.execute(plan, [s0])

check("Strong-tier failure — no further escalation",
      _test_strong_failure_no_escalation,
      all_status="failed",
      has_escalated=False)


# ======================================================================
# Test 11: Timeout → escalation
# ======================================================================

async def _test_timeout_escalation():
    s0 = _make_segment(0, "Question that times out on weak", "weak_model")
    plan = [_make_plan_step(0, [s0])]
    backend = MockLLMBackend(timeout_ids={0}, strong_delay=0.02)
    engine = ParallelExecutionEngine(
        backend=backend,
        enable_escalation=True,
        default_timeout=0.05,
        escalation_timeout=5.0,
    )
    return await engine.execute(plan, [s0])

check("Timeout → escalation to strong",
      _test_timeout_escalation,
      has_status="escalated",
      has_escalated=True)


# ======================================================================
# Test 12: verify_required routes to strong
# ======================================================================

async def _test_verify_required():
    s0 = _make_segment(0, "Ambiguous question", "verify_required",
                        intent="reasoning", complexity="medium")
    plan = [_make_plan_step(0, [s0])]
    backend = MockLLMBackend(strong_delay=0.02)
    engine = ParallelExecutionEngine(backend=backend)
    output = await engine.execute(plan, [s0])

    # Should use strong model
    assert "70b" in output["results"][0]["model_used"], (
        f"verify_required should use strong model, "
        f"got {output['results'][0]['model_used']}"
    )
    return output

check("verify_required → routes to strong model",
      _test_verify_required,
      all_status="success")


# ======================================================================
# Test 13: Weak vs Strong model selection
# ======================================================================

async def _test_model_selection():
    s_weak = _make_segment(0, "Simple question", "weak_model")
    s_strong = _make_segment(1, "Hard question", "strong_model",
                              intent="code", complexity="hard")
    plan = [_make_plan_step(0, [s_weak, s_strong], mode="parallel")]
    backend = MockLLMBackend(weak_delay=0.02, strong_delay=0.03)
    engine = ParallelExecutionEngine(backend=backend)
    output = await engine.execute(plan, [s_weak, s_strong])

    r0 = output["results"][0]
    r1 = output["results"][1]
    assert "8b" in r0["model_used"], f"Weak should use 8b, got {r0['model_used']}"
    assert "70b" in r1["model_used"], f"Strong should use 70b, got {r1['model_used']}"
    return output

check("Weak vs Strong — correct model selection",
      _test_model_selection,
      all_status="success")


# ======================================================================
# Test 14: Output ordering — results sorted by segment_id
# ======================================================================

async def _test_output_ordering():
    segs = [
        _make_segment(i, f"Question {i}", "weak_model")
        for i in range(5)
    ]
    plan = [_make_plan_step(0, segs, mode="parallel")]
    backend = MockLLMBackend(weak_delay=0.02)
    engine = ParallelExecutionEngine(backend=backend)
    output = await engine.execute(plan, segs)

    ids = [r["segment_id"] for r in output["results"]]
    assert ids == sorted(ids), f"Results not sorted: {ids}"
    return output

check("Output ordering — sorted by segment_id",
      _test_output_ordering,
      min_results=5,
      all_status="success")


# ======================================================================
# Test 15: Timing fields populated
# ======================================================================

async def _test_timing_fields():
    s0 = _make_segment(0, "Timed question", "weak_model")
    plan = [_make_plan_step(0, [s0])]
    backend = MockLLMBackend(weak_delay=0.03)
    engine = ParallelExecutionEngine(backend=backend)
    output = await engine.execute(plan, [s0])

    t = output["results"][0]["timing"]
    assert t["dispatch_time"] is not None, "dispatch_time missing"
    assert t["start_time"] is not None, "start_time missing"
    assert t["first_token_time"] is not None, "first_token_time missing"
    assert t["end_time"] is not None, "end_time missing"
    assert t["latency_ms"] > 0, f"latency_ms should be > 0, got {t['latency_ms']}"
    assert t["start_time"] >= t["dispatch_time"], "start must be >= dispatch"
    assert t["first_token_time"] >= t["start_time"], "first_token >= start"
    assert t["end_time"] >= t["first_token_time"], "end >= first_token"
    return output

check("Timing fields — all populated and ordered",
      _test_timing_fields,
      all_status="success",
      all_have_timing=True)


# ======================================================================
# Test 16: Aggregate statistics
# ======================================================================

async def _test_aggregate_stats():
    s0 = _make_segment(0, "Normal question", "weak_model")
    s1 = _make_segment(1, "Blocked content", "safe_block")
    s2 = _make_segment(2, "Another normal", "strong_model",
                        intent="code", complexity="hard")
    plan = [_make_plan_step(0, [s0, s1, s2], mode="parallel")]
    backend = MockLLMBackend(weak_delay=0.02, strong_delay=0.03)
    engine = ParallelExecutionEngine(backend=backend)
    output = await engine.execute(plan, [s0, s1, s2])

    stats = output["stats"]
    assert stats["total_segments"] == 3, f"Expected 3 total, got {stats['total_segments']}"
    assert stats["success"] == 2, f"Expected 2 success, got {stats['success']}"
    assert stats["blocked"] == 1, f"Expected 1 blocked, got {stats['blocked']}"
    assert stats["weak_calls"] == 1, f"Expected 1 weak call, got {stats['weak_calls']}"
    assert stats["strong_calls"] == 1, f"Expected 1 strong call, got {stats['strong_calls']}"
    return output

check("Aggregate statistics — counts match",
      _test_aggregate_stats,
      min_results=3)


# ======================================================================
# Test 17: Context injection for dependent segments
# ======================================================================

async def _test_context_injection():
    s0 = _make_segment(0, "Find x in 3x = 9", "strong_model",
                        intent="math", complexity="hard")
    s1 = _make_segment(1, "Use x to compute x + 5", "strong_model",
                        intent="math", complexity="hard", depends_on=[0])
    plan = [
        _make_plan_step(0, [s0]),
        _make_plan_step(1, [s1]),
    ]
    backend = MockLLMBackend(strong_delay=0.02)
    engine = ParallelExecutionEngine(backend=backend)
    output = await engine.execute(plan, [s0, s1])

    # Check that the second call received context from the first
    assert len(backend.call_log) == 2, f"Expected 2 calls, got {len(backend.call_log)}"
    second_call = backend.call_log[1]
    assert second_call["context"] is not None, (
        "Second segment should receive context from first"
    )
    return output

check("Context injection — dependent gets upstream response",
      _test_context_injection,
      all_status="success")


# ======================================================================
# Test 18: Independent segments get no context
# ======================================================================

async def _test_no_context_for_independent():
    s0 = _make_segment(0, "Independent Q1", "weak_model")
    s1 = _make_segment(1, "Independent Q2", "weak_model")
    plan = [_make_plan_step(0, [s0, s1], mode="parallel")]
    backend = MockLLMBackend(weak_delay=0.02)
    engine = ParallelExecutionEngine(backend=backend)
    output = await engine.execute(plan, [s0, s1])

    for call in backend.call_log:
        assert call["context"] is None, (
            f"Segment [{call['segment_id']}]: independent should have "
            f"no context, got {call['context']}"
        )
    return output

check("No context for independent segments",
      _test_no_context_for_independent,
      all_status="success")


# ======================================================================
# Test 19: Empty plan — no crash
# ======================================================================

async def _test_empty_plan():
    backend = MockLLMBackend()
    engine = ParallelExecutionEngine(backend=backend)
    return await engine.execute([], [])

check("Empty plan — graceful empty result",
      _test_empty_plan,
      min_results=0)


# ======================================================================
# Test 20: Large parallel batch — 10 segments
# ======================================================================

async def _test_large_batch():
    segs = [
        _make_segment(i, f"Question {i}", "weak_model")
        for i in range(10)
    ]
    plan = [_make_plan_step(0, segs, mode="parallel")]
    backend = MockLLMBackend(weak_delay=0.03)
    engine = ParallelExecutionEngine(backend=backend)
    return await engine.execute(plan, segs)

check("Large parallel batch — 10 segments",
      _test_large_batch,
      min_results=10,
      all_status="success",
      min_speedup=3.0)


# ======================================================================
# Test 21: Mixed failures — some succeed, some fail
# ======================================================================

async def _test_mixed_failures():
    segs = [
        _make_segment(0, "Normal Q0", "weak_model"),
        _make_segment(1, "Fail Q1", "weak_model"),
        _make_segment(2, "Normal Q2", "weak_model"),
        _make_segment(3, "Fail Q3", "weak_model"),
    ]
    plan = [_make_plan_step(0, segs, mode="parallel")]
    backend = MockLLMBackend(
        weak_delay=0.02,
        strong_delay=0.03,
        failure_ids={1, 3},
    )
    engine = ParallelExecutionEngine(backend=backend, enable_escalation=True)
    output = await engine.execute(plan, segs)

    # Segments 0 & 2 succeed, 1 & 3 escalate
    return output

check("Mixed failures — isolated per segment",
      _test_mixed_failures,
      min_results=4,
      specific_statuses={0: "success", 1: "escalated",
                         2: "success", 3: "escalated"})


# ======================================================================
# Test 22: Blocked segment latency is zero
# ======================================================================

async def _test_blocked_latency():
    s0 = _make_segment(0, "Blocked content", "safe_block")
    plan = [_make_plan_step(0, [s0])]
    backend = MockLLMBackend()
    engine = ParallelExecutionEngine(backend=backend)
    output = await engine.execute(plan, [s0])

    lat = output["results"][0]["timing"]["latency_ms"]
    assert lat == 0.0, f"Blocked latency should be 0, got {lat}"
    return output

check("Blocked segment — zero latency",
      _test_blocked_latency,
      all_status="blocked")


# ======================================================================
# Test 23: execute_sync wrapper
# ======================================================================

def _test_sync_wrapper():
    s0 = _make_segment(0, "Sync test question", "weak_model")
    plan = [_make_plan_step(0, [s0])]
    backend = MockLLMBackend(weak_delay=0.02)
    engine = ParallelExecutionEngine(backend=backend)
    return engine.execute_sync(plan, [s0])

print("\n" + "=" * 65)
print("TEST: execute_sync — synchronous wrapper")
print("=" * 65)

output_23 = _test_sync_wrapper()
if output_23["results"][0]["status"] == "success":
    PASS_COUNT += 1
    print(f"  Segments: {len(output_23['results'])}")
    print("✅ PASSED")
else:
    FAIL_COUNT += 1
    print("❌ FAILED: sync wrapper did not succeed")


# ======================================================================
# Test 24: Multi-step plan with 3 steps
# ======================================================================

async def _test_three_step_plan():
    """
    Step 0: segments 0 & 1 (parallel)
    Step 1: segment 2 depends on 0
    Step 2: segment 3 depends on 1 & 2
    """
    s0 = _make_segment(0, "Independent Q0", "weak_model")
    s1 = _make_segment(1, "Independent Q1", "weak_model")
    s2 = _make_segment(2, "Depends on 0", "strong_model",
                        intent="reasoning", complexity="hard",
                        depends_on=[0])
    s3 = _make_segment(3, "Depends on 1 and 2", "strong_model",
                        intent="reasoning", complexity="hard",
                        depends_on=[1, 2])
    plan = [
        _make_plan_step(0, [s0, s1], mode="parallel"),
        _make_plan_step(1, [s2]),
        _make_plan_step(2, [s3]),
    ]
    backend = MockLLMBackend(weak_delay=0.02, strong_delay=0.03)
    engine = ParallelExecutionEngine(backend=backend)
    output = await engine.execute(plan, [s0, s1, s2, s3])

    # Check context was injected for s3 (depends on both s1 and s2)
    # The call_log should show context for s2 (depends on s0)
    # and context for s3 (depends on s1 and s2)
    s2_call = backend.call_log[2]  # third call
    s3_call = backend.call_log[3]  # fourth call
    assert s2_call["context"] is not None, "s2 should get context from s0"
    assert s3_call["context"] is not None, "s3 should get context from s1 and s2"

    return output

check("Three-step plan — multi-level dependencies",
      _test_three_step_plan,
      min_results=4,
      all_status="success")


# ======================================================================
# Test 25: _empty_result helper
# ======================================================================

print("\n" + "=" * 65)
print("TEST: _empty_result helper")
print("=" * 65)

seg = {"segment_id": 42, "text": "hello", "route_tier": "weak_model",
       "intent_label": "retrieval", "complexity_band": "simple"}
er = _empty_result(seg)
errors_25 = []
if er["segment_id"] != 42:
    errors_25.append(f"segment_id wrong: {er['segment_id']}")
if er["status"] != "pending":
    errors_25.append(f"status should be pending: {er['status']}")
if er["timing"]["dispatch_time"] is not None:
    errors_25.append("dispatch_time should be None initially")

if errors_25:
    FAIL_COUNT += 1
    print("❌ FAILED:")
    for e in errors_25:
        print(f"  {e}")
else:
    PASS_COUNT += 1
    print("✅ PASSED")


# ======================================================================
# Test 26: print_results — does not crash
# ======================================================================

async def _test_print_results():
    s0 = _make_segment(0, "Print test", "weak_model")
    s1 = _make_segment(1, "Blocked print", "safe_block")
    plan = [_make_plan_step(0, [s0, s1], mode="parallel")]
    backend = MockLLMBackend(weak_delay=0.02)
    engine = ParallelExecutionEngine(backend=backend)
    output = await engine.execute(plan, [s0, s1])
    engine.print_results(output, show_responses=True)
    return output

check("print_results — no crash",
      _test_print_results,
      min_results=2)


# ======================================================================
# Summary
# ======================================================================

print("\n\n" + "=" * 65)
print(f"MODULE 5 STANDALONE RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed "
      f"(out of {PASS_COUNT + FAIL_COUNT})")
print("=" * 65)

if FAIL_COUNT > 0:
    sys.exit(1)
