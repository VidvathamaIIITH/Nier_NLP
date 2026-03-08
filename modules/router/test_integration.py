"""
Module 1 → Module 2 → Module 3 → Module 4  Integration Test
==============================================================

Pipes real prompts through the full four-module pipeline:
  1. Semantic Decomposer   →  segments
  2. Dependency Graph      →  DAG (nodes, edges, plan)
  3. Intent & Complexity   →  annotated nodes
  4. Learning-Based Router →  routed nodes with tier assignments

Run from the repo root:
    PYTHONPATH=. python modules/router/test_integration.py
"""

import sys, os
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

decomposer = SemanticDecomposer()
graph_builder = DependencyGraphBuilder()
estimator = IntentComplexityEstimator()
router = LearningBasedRouter(mode="heuristic")

PASS_COUNT = 0
FAIL_COUNT = 0


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def run_pipeline(label: str, prompt: str,
                 expected_tiers: list = None,
                 min_segments: int = None,
                 has_weak: bool = None,
                 has_strong: bool = None,
                 has_block: bool = None,
                 has_verify: bool = None):
    """Decompose → Graph → Annotate → Route → Print → Assert."""
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

    # --- Module 4 ---
    routed = router.route_all(annotated)
    print(f"\n  Module 4 → routing decisions:")
    for seg in routed:
        sid = seg.get("segment_id", "?")
        tier = seg["route_tier"]
        conf = seg["route_confidence"]
        method = seg["route_method"]
        icon = {"weak_model": "🟢", "strong_model": "🔴",
                "safe_block": "🚫", "verify_required": "⚠️"}.get(tier, "❓")
        print(f"    [{sid}] {icon} {tier:<16} "
              f"(conf={conf:.2f}, via {method})")

    # Print cost stats
    stats = router.compute_stats(routed)
    print(f"\n  Stats: weak={stats['weak_count']}, strong={stats['strong_count']}, "
          f"block={stats['block_count']}, verify={stats['verify_count']} "
          f"| savings={stats['estimated_savings_pct']}%")

    # --- Assertions ---
    errors = []

    if min_segments is not None and len(routed) < min_segments:
        errors.append(
            f"Expected >= {min_segments} segments, got {len(routed)}"
        )

    if expected_tiers is not None:
        for i, exp in enumerate(expected_tiers):
            if i < len(routed) and routed[i]["route_tier"] != exp:
                errors.append(
                    f"Segment [{i}]: expected tier '{exp}', "
                    f"got '{routed[i]['route_tier']}'"
                )

    if has_weak is not None:
        any_weak = any(s["route_tier"] == ROUTE_WEAK for s in routed)
        if has_weak and not any_weak:
            errors.append("Expected at least one weak-routed segment")
        if not has_weak and any_weak:
            errors.append("Did not expect any weak-routed segments")

    if has_strong is not None:
        any_strong = any(s["route_tier"] == ROUTE_STRONG for s in routed)
        if has_strong and not any_strong:
            errors.append("Expected at least one strong-routed segment")
        if not has_strong and any_strong:
            errors.append("Did not expect any strong-routed segments")

    if has_block is not None:
        any_block = any(s["route_tier"] == ROUTE_BLOCK for s in routed)
        if has_block and not any_block:
            errors.append("Expected at least one blocked segment")
        if not has_block and any_block:
            errors.append("Did not expect any blocked segments")

    if has_verify is not None:
        any_verify = any(s["route_tier"] == ROUTE_VERIFY for s in routed)
        if has_verify and not any_verify:
            errors.append("Expected at least one verify-required segment")

    # Every routed segment must have all required fields
    for seg in routed:
        for key in ["route_tier", "route_confidence", "route_reason",
                     "route_method", "intent_label", "complexity_band"]:
            if key not in seg:
                errors.append(f"Segment [{seg.get('segment_id', '?')}] missing '{key}'")

    if errors:
        FAIL_COUNT += 1
        print("❌ FAILED:")
        for e in errors:
            print(f"  {e}")
    else:
        PASS_COUNT += 1
        print("✅ PASSED")

    return routed


# ======================================================================
# Integration Tests
# ======================================================================

# 1. Mixed intent: retrieval (weak) + math (strong)
run_pipeline(
    "Mixed intent: retrieval + math",
    "Who is the CEO of Google and solve 2x + 4 = 10",
    has_weak=True,
    has_strong=True,
    min_segments=2,
)

# 2. Dependent chain: find → use result
run_pipeline(
    "Dependent chain: find X then use result",
    "Find the value of x in 3x + 6 = 18, then use that result to calculate 2x + 5.",
    has_strong=True,
    min_segments=2,
)

# 3. Simple factual retrieval → weak
run_pipeline(
    "Simple factual retrieval → all weak",
    "What is the capital of France?",
    has_weak=True,
    min_segments=1,
)

# 4. Creative generation (simple) → weak
run_pipeline(
    "Creative haiku → weak",
    "Write a haiku about rain.",
    has_weak=True,
    min_segments=1,
)

# 5. Code generation → strong
run_pipeline(
    "Code generation → strong",
    "Write a Python function that implements binary search.",
    has_strong=True,
    min_segments=1,
)

# 6. Three independent tasks: mix of weak and strong
run_pipeline(
    "Three independent: translate + name + code",
    "Translate 'hello world' to Spanish. Name the largest ocean. Write a quicksort in Python.",
    has_strong=True,
    min_segments=2,
)

# 7. Code block passthrough → strong
run_pipeline(
    "Code block passthrough → strong",
    'Here is my code: ```python\ndef foo():\n    return 42\n``` Explain what it does.',
    has_strong=True,
    min_segments=1,
)

# 8. Describe + dependent explanation
run_pipeline(
    "Describe + dependent explain → mixed routing",
    "Describe photosynthesis. Then explain why it is essential for life on Earth.",
    min_segments=2,
)

# 9. Math word problem → strong
run_pipeline(
    "Math word problem → strong",
    "Miss Adamson has four classes, each with 20 students. She makes a study guide "
    "of 7 pages for each student. How many pages does she make in total?",
    has_strong=True,
)

# 10. Chitchat greeting → weak
run_pipeline(
    "Simple chitchat → weak",
    "Hello! Good morning!",
    has_weak=True,
)

# 11. Classification task → weak (simple)
run_pipeline(
    "Classification → weak",
    "Classify the following as vegetable or fruit: Apple, Broccoli, Carrot.",
    has_weak=True,
)

# 12. Non-English prompt
run_pipeline(
    "Non-English Spanish → routing decision",
    "¿Qué diferencia al café cappuccino del café latte?",
    min_segments=1,
)

# 13. Hard multi-step reasoning → strong
run_pipeline(
    "Multi-step reasoning → strong",
    "First, identify the logical fallacy in the argument. "
    "Second, explain why it is a fallacy. "
    "Third, provide a corrected version of the argument. "
    "Finally, discuss the broader implications.",
    has_strong=True,
)

# 14. Compare and contrast → should stay together, route appropriately
run_pipeline(
    "Compare and contrast → single segment routing",
    "Compare and contrast the tourist attractions of Paris and Tokyo.",
    min_segments=1,
)

# 15. Phase B: train on pipeline outputs, then re-route
print("\n\n" + "#" * 70)
print("INTEGRATION TEST: Phase B — train learned router on pipeline outputs")
print("#" * 70)

# Collect training data from a diverse batch of prompts
# We use many prompts to give the learned router enough signal
training_prompts = [
    # Weak-eligible
    "What is the capital of France?",
    "Hello, how are you?",
    "Who is the president of the United States?",
    "What is the boiling point of water?",
    "Hi there!",
    "Good morning!",
    "Where is the Eiffel Tower?",
    "Summarize this article.",
    "Classify this as positive or negative.",
    "Rewrite this sentence more formally.",
    "Write a haiku about winter.",
    "Translate 'good morning' to Japanese.",
    "Name the largest ocean.",
    "Tell me the population of India.",
    "Define photosynthesis.",
    # Strong-eligible
    "Solve 3x + 9 = 21",
    "Write a Python merge sort",
    "Given that all birds have wings, and a penguin is a bird, can it fly?",
    "Explain the theory of relativity.",
    "What is 5 factorial?",
    "Implement binary search in Java.",
    "Derive the quadratic formula step by step.",
    "Write a recursive Fibonacci function in Python.",
    "If all A are B, and all B are C, what can we conclude?",
    "Calculate the integral of x^2 from 0 to 3.",
    "First compute the derivative, then integrate the result.",
    "Debug this code and explain the fix.",
    "Given these premises, prove the theorem using induction.",
    "Write a Python class for a binary search tree with insert and delete.",
    "Solve the system of equations: 2x + y = 5, x - y = 1.",
]

all_annotated = []
for p in training_prompts:
    segs = decomposer.decompose(p)
    g = graph_builder.build(segs)
    ann = estimator.estimate_graph(g)
    all_annotated.extend(ann)

# Duplicate to increase training set size
import numpy as np
from modules.router.router import extract_features

X_train, y_train = router.generate_training_data(all_annotated * 3)

# Train a learned router
learned_router = LearningBasedRouter(mode="learned")
stats = learned_router.train(X_train, y_train, learning_rate=0.5, epochs=500)

print(f"\n  Training samples:  {stats['n_samples']}")
print(f"  Training accuracy: {stats['accuracy']:.2%}")
print(f"  Training loss:     {stats['loss']:.4f}")

# Re-route the same segments with the learned router
learned_routed = learned_router.route_all(all_annotated)

# Compare heuristic vs learned
heuristic_routed = router.route_all(all_annotated)
agreement = sum(
    1 for h, l in zip(heuristic_routed, learned_routed)
    if h["route_tier"] == l["route_tier"]
)
total = len(heuristic_routed)
agreement_pct = agreement / total * 100

print(f"  Heuristic-Learned agreement: {agreement}/{total} ({agreement_pct:.1f}%)")

h_stats = router.compute_stats(heuristic_routed)
l_stats = learned_router.compute_stats(learned_routed)
print(f"  Heuristic: weak={h_stats['weak_count']}, strong={h_stats['strong_count']}, "
      f"savings={h_stats['estimated_savings_pct']}%")
print(f"  Learned:   weak={l_stats['weak_count']}, strong={l_stats['strong_count']}, "
      f"savings={l_stats['estimated_savings_pct']}%")

errors_15 = []
if stats["accuracy"] < 0.65:
    errors_15.append(f"Training accuracy {stats['accuracy']:.2%} < 65%")
if agreement_pct < 50:
    errors_15.append(f"Agreement {agreement_pct:.1f}% < 50%")

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
print(f"MODULE 4 INTEGRATION RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed "
      f"(out of {PASS_COUNT + FAIL_COUNT})")
print("=" * 70)

if FAIL_COUNT > 0:
    sys.exit(1)
