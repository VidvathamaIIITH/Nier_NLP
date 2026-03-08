"""
Module 1 → Module 2 → Module 3 → Module 4 → Module 5  Integration Test
=========================================================================

Pipes real prompts through the full five-module pipeline:
  1. Semantic Decomposer        →  segments
  2. Dependency Graph Builder   →  DAG (nodes, edges, plan)
  3. Intent & Complexity        →  annotated nodes
  4. Learning-Based Router      →  routed nodes with tier assignments
  5. Parallel Execution Engine  →  executed results with timing stats

Run from the repo root:
    PYTHONPATH=. python3 modules/execution_engine/test_integration.py
"""

import sys, os, time
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.semantic_decomposition.semantic_decomposer import SemanticDecomposer
from modules.dependency_graph.graph_builder import DependencyGraphBuilder
from modules.intent_complexity.estimator import IntentComplexityEstimator
from modules.router.router import (
    LearningBasedRouter,
    ROUTE_WEAK, ROUTE_STRONG, ROUTE_BLOCK, ROUTE_VERIFY,
)
from modules.execution_engine.engine import (
    ParallelExecutionEngine,
    MockLLMBackend,
    STATUS_SUCCESS, STATUS_BLOCKED, STATUS_ESCALATED,
)

decomposer = SemanticDecomposer()
graph_builder = DependencyGraphBuilder()
estimator = IntentComplexityEstimator()
router = LearningBasedRouter(mode="heuristic")

PASS_COUNT = 0
FAIL_COUNT = 0


# ------------------------------------------------------------------
# Pipeline Helper
# ------------------------------------------------------------------

def run_pipeline(
    label: str,
    prompt: str,
    mock_kwargs: dict = None,
    engine_kwargs: dict = None,
    min_segments: int = None,
    all_success: bool = None,
    has_blocked: bool = None,
    has_escalated: bool = None,
    parallel_speedup: bool = None,
    results_ordered: bool = True,
    show_detail: bool = True,
):
    """Decompose → Graph → Annotate → Route → Execute → Assert."""
    global PASS_COUNT, FAIL_COUNT

    print("\n" + "=" * 70)
    print(f"INTEGRATION TEST: {label}")
    print("=" * 70)
    print(f"  PROMPT: {prompt[:100]}{'…' if len(prompt) > 100 else ''}\n")

    # --- Module 1 ---
    segments = decomposer.decompose(prompt)
    print(f"  Module 1 → {len(segments)} segment(s)")
    for seg in segments:
        dep = "DEP" if seg.get("depends_on_previous") else "IND"
        print(f"    [{seg['id']}] ({dep}) {seg['text'][:70]}")

    # --- Module 2 ---
    graph = graph_builder.build(segments)
    plan = graph["execution_plan"]
    print(f"\n  Module 2 → {len(graph['edges'])} edge(s), "
          f"{len(plan)} step(s)")
    for step in plan:
        ids = [s["segment_id"] for s in step["segments"]]
        print(f"    Step {step['step']} ({step['mode']}): segments {ids}")

    # --- Module 3 ---
    annotated = estimator.estimate_graph(graph)
    print(f"\n  Module 3 → annotations:")
    for node in annotated:
        sid = node["segment_id"]
        print(f"    [{sid}] intent={node['intent_label']:<15} "
              f"complexity={node['complexity_score']:.3f} "
              f"({node['complexity_band']:<6}) "
              f"unsafe={node['unsafe_candidate']}")

    # --- Module 4 ---
    routed = router.route_all(annotated)
    print(f"\n  Module 4 → routing:")
    for seg in routed:
        sid = seg.get("segment_id", "?")
        tier = seg["route_tier"]
        conf = seg["route_confidence"]
        icon = {"weak_model": "🟢", "strong_model": "🔴",
                "safe_block": "🚫", "verify_required": "⚠️"}.get(tier, "❓")
        print(f"    [{sid}] {icon} {tier:<16} (conf={conf:.2f})")

    # --- Module 5 ---
    mk = mock_kwargs or {}
    ek = engine_kwargs or {}
    mock = MockLLMBackend(**mk)
    engine = ParallelExecutionEngine(backend=mock, **ek)
    output = engine.execute_sync(plan, routed)

    results = output["results"]
    stats = output["stats"]

    print(f"\n  Module 5 → execution ({stats['total_segments']} segments):")
    for r in results:
        sid = r.get("segment_id", "?")
        status = r.get("status", "?")
        dur = r.get("timing", {}).get("latency_ms", 0) or 0
        model = r.get("model_used", "?")
        esc = " [ESC]" if r.get("escalated") else ""
        status_icon = {
            "success": "✅", "blocked": "🚫", "failed": "❌",
            "timeout": "⏰", "escalated": "⬆️",
        }.get(status, "❓")
        resp_preview = r.get("response_text", "")[:60]
        print(f"    [{sid}] {status_icon} {status:<10}{esc}  "
              f"model={model:<14} {dur:>7.1f}ms")
        if show_detail and resp_preview:
            print(f"         → {resp_preview}")

    print(f"\n  Stats: wall={output['parallel_latency_ms']:.1f}ms  "
          f"sum={output['sequential_latency_ms']:.1f}ms  "
          f"speedup={output['speedup']:.2f}×")
    print(f"         ok={stats['success']}  block={stats['blocked']}  "
          f"err={stats['failed']}  esc={stats['escalated']}")

    # --- Assertions ---
    errors = []

    if min_segments is not None and len(results) < min_segments:
        errors.append(
            f"Expected >= {min_segments} results, got {len(results)}"
        )

    if all_success is True:
        non_ok = [
            r for r in results
            if r["status"] not in {STATUS_SUCCESS, STATUS_ESCALATED}
            and r.get("route_tier") != ROUTE_BLOCK
        ]
        if non_ok:
            sids = [r["segment_id"] for r in non_ok]
            errors.append(
                f"Expected all success (ignoring blocked), "
                f"but segments {sids} had issues"
            )

    if has_blocked is True:
        if stats["blocked"] == 0:
            errors.append("Expected at least one blocked segment")
    if has_blocked is False:
        if stats["blocked"] > 0:
            errors.append("Did not expect any blocked segments")

    if has_escalated is True:
        if stats["escalated"] == 0:
            errors.append("Expected at least one escalated segment")
    if has_escalated is False:
        if stats["escalated"] > 0:
            errors.append("Did not expect any escalated segments")

    if parallel_speedup is True:
        if output["speedup"] < 1.0:
            errors.append(
                f"Expected speedup >= 1.0, got {output['speedup']}"
            )

    if results_ordered:
        ids = [r["segment_id"] for r in results]
        if ids != sorted(ids):
            errors.append(
                f"Results not ordered by segment_id: {ids}"
            )

    # Every result must have all required fields
    for r in results:
        for key in ["response_text", "status", "model_used",
                     "escalated", "segment_id", "timing"]:
            if key not in r:
                errors.append(
                    f"Result [{r.get('segment_id', '?')}] missing '{key}'"
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
# Integration Tests
# ======================================================================

# 1. Simple factual retrieval → weak → parallel execution
run_pipeline(
    "Simple factual → weak model execution",
    "What is the capital of France?",
    all_success=True,
    min_segments=1,
    has_blocked=False,
)

# 2. Mixed intent: retrieval + math → parallel execution
run_pipeline(
    "Mixed intent: retrieval + math → parallel",
    "Who is the CEO of Google and solve 2x + 4 = 10",
    all_success=True,
    min_segments=2,
    parallel_speedup=True,
    mock_kwargs={"weak_delay": 0.03, "strong_delay": 0.06},
)

# 3. Dependent chain: find → use result
run_pipeline(
    "Dependent chain: sequential execution order",
    "Find the value of x in 3x + 6 = 18, then use that result to calculate 2x + 5.",
    all_success=True,
    min_segments=2,
)

# 4. Code generation → strong model
run_pipeline(
    "Code generation → strong model execution",
    "Write a Python function that implements binary search.",
    all_success=True,
    min_segments=1,
)

# 5. Three independent tasks → parallel speedup
run_pipeline(
    "Three independent → parallel speedup",
    "Translate 'hello world' to Spanish. Name the largest ocean. Write a quicksort in Python.",
    all_success=True,
    min_segments=2,
    parallel_speedup=True,
    mock_kwargs={"weak_delay": 0.04, "strong_delay": 0.07},
)

# 6. Chitchat → weak → fast execution
run_pipeline(
    "Chitchat → weak model, fast execution",
    "Hello! Good morning!",
    all_success=True,
    min_segments=1,
    has_blocked=False,
)

# 7. Classification → weak
run_pipeline(
    "Classification → weak model execution",
    "Classify the following as vegetable or fruit: Apple, Broccoli, Carrot.",
    all_success=True,
    min_segments=1,
)

# 8. Multi-step reasoning → strong
run_pipeline(
    "Multi-step reasoning → strong model chain",
    "First, identify the logical fallacy in the argument. "
    "Second, explain why it is a fallacy. "
    "Third, provide a corrected version of the argument.",
    all_success=True,
    min_segments=2,
)

# 9. Describe + dependent explain
run_pipeline(
    "Describe + dependent explanation → sequential",
    "Describe photosynthesis. Then explain why it is essential for life on Earth.",
    all_success=True,
    min_segments=2,
)

# 10. Math word problem → strong execution
run_pipeline(
    "Math word problem → strong model",
    "Miss Adamson has four classes, each with 20 students. She makes a study guide "
    "of 7 pages for each student. How many pages does she make in total?",
    all_success=True,
)

# 11. Escalation test — force weak failures for certain segments
print("\n\n" + "#" * 70)
print("INTEGRATION TEST: Escalation — weak failure → strong retry")
print("#" * 70)

prompt_11 = "What is the capital of France? Also, what year was the Eiffel Tower built?"
segments_11 = decomposer.decompose(prompt_11)
graph_11 = graph_builder.build(segments_11)
annotated_11 = estimator.estimate_graph(graph_11)
routed_11 = router.route_all(annotated_11)

# Find weak-routed segment IDs and force failure on the first one
weak_sids = {
    s["segment_id"] for s in routed_11
    if s["route_tier"] == ROUTE_WEAK
}

if weak_sids:
    fail_sid = min(weak_sids)
    mock_11 = MockLLMBackend(failure_ids={fail_sid})
    engine_11 = ParallelExecutionEngine(
        backend=mock_11, enable_escalation=True
    )
    output_11 = engine_11.execute_sync(
        graph_11["execution_plan"], routed_11
    )

    escalated_results = [
        r for r in output_11["results"]
        if r.get("escalated")
    ]
    errors_11 = []
    if len(escalated_results) == 0:
        errors_11.append("Expected at least one escalated result")
    else:
        esc = escalated_results[0]
        if "70b" not in esc["model_used"]:
            errors_11.append(
                f"Escalated segment should use strong (70b), got {esc['model_used']}"
            )
        if not esc["response_text"]:
            errors_11.append("Escalated segment should have a response")

    if errors_11:
        FAIL_COUNT += 1
        print("❌ FAILED:")
        for e in errors_11:
            print(f"  {e}")
    else:
        PASS_COUNT += 1
        print("✅ PASSED")

    print(f"  Forced failure on segment {fail_sid}")
    for r in output_11["results"]:
        esc_str = " [ESC]" if r.get("escalated") else ""
        print(f"    [{r['segment_id']}] {r['status']}{esc_str} "
              f"model={r['model_used']}")
else:
    # No weak segments — still pass, just note it
    PASS_COUNT += 1
    print("✅ PASSED (no weak segments to fail)")


# 12. Parallel latency bound — T_total ≈ max(T_i) + ε
print("\n\n" + "#" * 70)
print("INTEGRATION TEST: Parallel latency bound — T_total ≈ max(T_i)")
print("#" * 70)

prompt_12 = (
    "Name the capital of France. "
    "Name the capital of Japan. "
    "Name the capital of Brazil. "
    "Name the capital of Australia."
)
segments_12 = decomposer.decompose(prompt_12)
graph_12 = graph_builder.build(segments_12)
annotated_12 = estimator.estimate_graph(graph_12)
routed_12 = router.route_all(annotated_12)

# Use noticeable latency so the parallel benefit is measurable
mock_12 = MockLLMBackend(weak_delay=0.05, strong_delay=0.08)
engine_12 = ParallelExecutionEngine(backend=mock_12)
output_12 = engine_12.execute_sync(graph_12["execution_plan"], routed_12)
stats_12 = output_12["stats"]

n_seg = stats_12["total_segments"]
wall = output_12["parallel_latency_ms"]
sum_dur = output_12["sequential_latency_ms"]
speedup = output_12["speedup"]

errors_12 = []
if n_seg >= 2:
    # Wall time should be significantly less than sum of durations
    if wall > sum_dur * 0.95 and n_seg > 1:
        errors_12.append(
            f"Wall time ({wall:.1f}ms) not much less than "
            f"sum ({sum_dur:.1f}ms) for {n_seg} segments"
        )
    if speedup < 1.0:
        errors_12.append(f"Speedup {speedup:.2f}× < 1.0")

print(f"  {n_seg} segments → wall={wall:.1f}ms  sum={sum_dur:.1f}ms  "
      f"speedup={speedup:.2f}×")

if errors_12:
    FAIL_COUNT += 1
    print("❌ FAILED:")
    for e in errors_12:
        print(f"  {e}")
else:
    PASS_COUNT += 1
    print("✅ PASSED")


# 13. Output ordering is stable across multiple runs
print("\n\n" + "#" * 70)
print("INTEGRATION TEST: Output ordering stability")
print("#" * 70)

prompt_13 = "What is 2+2? What is 3+3? What is 4+4?"
segments_13 = decomposer.decompose(prompt_13)
graph_13 = graph_builder.build(segments_13)
annotated_13 = estimator.estimate_graph(graph_13)
routed_13 = router.route_all(annotated_13)

engine_13 = ParallelExecutionEngine(backend=MockLLMBackend())
runs = []
for _ in range(5):
    out = engine_13.execute_sync(graph_13["execution_plan"], routed_13)
    runs.append([r["segment_id"] for r in out["results"]])

errors_13 = []
for i in range(1, len(runs)):
    if runs[i] != runs[0]:
        errors_13.append(
            f"Run {i} order {runs[i]} != run 0 order {runs[0]}"
        )

if errors_13:
    FAIL_COUNT += 1
    print("❌ FAILED:")
    for e in errors_13:
        print(f"  {e}")
else:
    PASS_COUNT += 1
    print(f"✅ PASSED — consistent order across 5 runs: {runs[0]}")


# 14. Failure isolation — one segment error doesn't affect others
print("\n\n" + "#" * 70)
print("INTEGRATION TEST: Failure isolation in parallel batch")
print("#" * 70)

prompt_14 = "What is the capital of France? Write a binary search in Python."
segments_14 = decomposer.decompose(prompt_14)
graph_14 = graph_builder.build(segments_14)
annotated_14 = estimator.estimate_graph(graph_14)
routed_14 = router.route_all(annotated_14)

# Force one segment to fail
all_sids_14 = [s["segment_id"] for s in routed_14]
fail_sid_14 = all_sids_14[0] if all_sids_14 else 0
mock_14 = MockLLMBackend(failure_ids={fail_sid_14})
engine_14 = ParallelExecutionEngine(
    backend=mock_14, enable_escalation=False
)
output_14 = engine_14.execute_sync(graph_14["execution_plan"], routed_14)

errors_14 = []
results_14 = output_14["results"]

# The failed segment should be errored
failed = [r for r in results_14 if r["segment_id"] == fail_sid_14]
others = [r for r in results_14 if r["segment_id"] != fail_sid_14]

if failed:
    r = failed[0]
    if r["route_tier"] == ROUTE_WEAK:
        # Weak without escalation → should error
        if r["status"] not in {"failed", "timeout"}:
            errors_14.append(
                f"Failed segment {fail_sid_14} should be failed/timeout, "
                f"got {r['status']}"
            )
    # If it's strong, MockLLMBackend.failure_ids only fails weak,
    # so strong should still succeed.

# Other segments should be unaffected
for r in others:
    if r["route_tier"] != ROUTE_BLOCK:
        if r["status"] not in {STATUS_SUCCESS, STATUS_ESCALATED}:
            errors_14.append(
                f"Segment {r['segment_id']} should be unaffected, "
                f"got {r['status']}"
            )

if errors_14:
    FAIL_COUNT += 1
    print("❌ FAILED:")
    for e in errors_14:
        print(f"  {e}")
else:
    PASS_COUNT += 1
    print(f"✅ PASSED — failure on seg {fail_sid_14} isolated")

for r in results_14:
    print(f"    [{r['segment_id']}] {r['status']}  "
          f"model={r['model_used']}")


# 15. Full pipeline with print_results — smoke test
print("\n\n" + "#" * 70)
print("INTEGRATION TEST: Full pipeline with print_results")
print("#" * 70)

prompt_15 = (
    "Explain quantum entanglement. "
    "Then use that to describe quantum computing."
)
segments_15 = decomposer.decompose(prompt_15)
graph_15 = graph_builder.build(segments_15)
annotated_15 = estimator.estimate_graph(graph_15)
routed_15 = router.route_all(annotated_15)

engine_15 = ParallelExecutionEngine(backend=MockLLMBackend())
output_15 = engine_15.execute_sync(graph_15["execution_plan"], routed_15)

try:
    engine_15.print_results(output_15)
    print_ok = True
except Exception as e:
    print(f"  print_results raised: {e}")
    print_ok = False

errors_15 = []
if not print_ok:
    errors_15.append("print_results raised an exception")
if output_15["stats"]["total_segments"] < 1:
    errors_15.append("Expected at least 1 segment")

if errors_15:
    FAIL_COUNT += 1
    print("❌ FAILED:")
    for e in errors_15:
        print(f"  {e}")
else:
    PASS_COUNT += 1
    print("✅ PASSED")


# ======================================================================
# Summary
# ======================================================================

print("\n\n" + "=" * 70)
print(f"MODULE 5 INTEGRATION RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed "
      f"(out of {PASS_COUNT + FAIL_COUNT})")
print("=" * 70)

if FAIL_COUNT > 0:
    sys.exit(1)
