"""
Module 1 → Module 2 → Module 3  Integration Test
===================================================

Pipes real prompts through the full three-module pipeline:
  1. Semantic Decomposer   →  segments
  2. Dependency Graph      →  DAG (nodes, edges, plan)
  3. Intent & Complexity   →  annotated nodes

Run from the repo root:
    PYTHONPATH=. python modules/intent_complexity/test_integration.py
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.semantic_decomposition.semantic_decomposer import SemanticDecomposer
from modules.dependency_graph.graph_builder import DependencyGraphBuilder
from modules.intent_complexity.estimator import IntentComplexityEstimator


decomposer = SemanticDecomposer()
graph_builder = DependencyGraphBuilder()
estimator = IntentComplexityEstimator()


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def run_pipeline(label: str, prompt: str,
                 expected_intents: list = None,
                 expected_bands: list = None,
                 min_segments: int = None,
                 check_unsafe: list = None):
    """Decompose → Graph → Annotate → Print → Assert."""
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
    print(f"\n  Module 2 → {len(graph['edges'])} edge(s), "
          f"{len(graph['parallel_groups'])} parallel group(s)")
    for step in graph["execution_plan"]:
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

    # --- Assertions ---
    errors = []

    if min_segments is not None and len(annotated) < min_segments:
        errors.append(
            f"Expected >= {min_segments} segments, got {len(annotated)}"
        )

    if expected_intents is not None:
        for i, exp in enumerate(expected_intents):
            if i < len(annotated) and annotated[i]["intent_label"] != exp:
                errors.append(
                    f"Segment {i}: expected intent '{exp}', "
                    f"got '{annotated[i]['intent_label']}'"
                )

    if expected_bands is not None:
        for i, exp in enumerate(expected_bands):
            if i < len(annotated) and annotated[i]["complexity_band"] != exp:
                errors.append(
                    f"Segment {i}: expected band '{exp}', "
                    f"got '{annotated[i]['complexity_band']}'"
                )

    if check_unsafe is not None:
        for i, exp in enumerate(check_unsafe):
            if i < len(annotated) and annotated[i]["unsafe_candidate"] != exp:
                errors.append(
                    f"Segment {i}: expected unsafe={exp}, "
                    f"got {annotated[i]['unsafe_candidate']}"
                )

    if errors:
        print("\n  ❌ FAILED:")
        for e in errors:
            print(f"    {e}")
        return False
    else:
        print("\n  ✅ PASSED")
        return True


# ======================================================================
# Integration Test Cases
# ======================================================================

results = []

# 1. Mixed-intent: retrieval + math (parallel)
results.append(run_pipeline(
    label="Mixed-intent: retrieval + math",
    prompt="Who is the CEO of Google and solve 2x + 4 = 10",
    expected_intents=["retrieval", "math"],
    min_segments=2,
))

# 2. Dependent chain: find → use result
results.append(run_pipeline(
    label="Dependent chain (find → use result)",
    prompt="Find X, then use that result to calculate Y.",
    min_segments=2,
))

# 3. Single retrieval question
results.append(run_pipeline(
    label="Single retrieval question",
    prompt="What is the capital of France?",
    expected_intents=["retrieval"],
    expected_bands=["simple"],
    min_segments=1,
))

# 4. Creative generation
results.append(run_pipeline(
    label="Creative generation — haiku",
    prompt="Write a haiku about rain.",
    expected_intents=["generation"],
    min_segments=1,
))

# 5. Translation
results.append(run_pipeline(
    label="Translation request",
    prompt="Translate 'hello world' to French.",
    expected_intents=["translation"],
    min_segments=1,
))

# 6. Three independent instructions (generation + retrieval + generation)
results.append(run_pipeline(
    label="Three independent instructions",
    prompt="Translate 'hello' to French. Name three prime numbers. Write a haiku about rain.",
    min_segments=3,
))

# 7. Code block passthrough
results.append(run_pipeline(
    label="Code block passthrough",
    prompt='Here is some code: ```python\nprint("hello")\n``` Explain what it does.',
    min_segments=1,
))

# 8. Describe + dependent explanation
results.append(run_pipeline(
    label="Describe + dependent explanation",
    prompt="Describe photosynthesis. Then explain why it matters for global ecology.",
    min_segments=2,
))

# 9. Compare and contrast (no false split)
results.append(run_pipeline(
    label="Compare and contrast (no false split)",
    prompt="Compare and contrast two popular tourist attractions in your hometown.",
    min_segments=1,
))

# 10. Math word problem
results.append(run_pipeline(
    label="Math word problem",
    prompt=(
        "Miss Adamson has four classes with 20 students each. She makes a "
        "study guide for her class and uses 5 sheets of paper per student. "
        "How many sheets of paper will she use for all of her students?"
    ),
))

# 11. Non-English prompt
results.append(run_pipeline(
    label="Non-English prompt (Spanish)",
    prompt="¿Qué diferencia al café cappuccino del café normal?",
    min_segments=1,
))

# 12. Classification task
results.append(run_pipeline(
    label="Classification task",
    prompt="Classify the following items as a vegetable or a fruit: Apple, Broccoli",
    expected_intents=["classification"],
    min_segments=1,
))


# ======================================================================
# Summary
# ======================================================================
passed = sum(results)
total = len(results)
print("\n\n" + "=" * 70)
print(f"  INTEGRATION M1→M2→M3: {passed}/{total} passed, {total - passed} failed")
if passed == total:
    print("  🎉 ALL INTEGRATION TESTS PASSED 🎉")
else:
    print("  ⚠  Some tests failed — review output above")
print("=" * 70 + "\n")

sys.exit(0 if passed == total else 1)
