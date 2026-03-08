"""
Module 4 — Standalone Tests
=============================

These tests use **hardcoded** annotated segment dicts (same format
Module 3 produces) so the router can be validated completely
independently of Modules 1, 2 & 3.

Run from the repo root:
    PYTHONPATH=. python modules/router/test.py
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from modules.router.router import (
    LearningBasedRouter,
    extract_features,
    ROUTE_WEAK, ROUTE_STRONG, ROUTE_BLOCK, ROUTE_VERIFY,
    NUM_FEATURES,
)

router = LearningBasedRouter(mode="heuristic")

PASS_COUNT = 0
FAIL_COUNT = 0


def check(test_name: str, segment: dict,
          expected_tier: str = None,
          tier_in: list = None,
          min_confidence: float = None,
          max_confidence: float = None,
          expected_method: str = None):
    """Run one test case and assert expectations."""
    global PASS_COUNT, FAIL_COUNT

    print("\n" + "=" * 65)
    print(f"TEST: {test_name}")
    print("=" * 65)

    result = router.route(segment)
    router.print_routes([result], show_reason=True)

    errors = []

    if expected_tier is not None and result["route_tier"] != expected_tier:
        errors.append(
            f"  TIER: expected '{expected_tier}', got '{result['route_tier']}'"
        )

    if tier_in is not None and result["route_tier"] not in tier_in:
        errors.append(
            f"  TIER: expected one of {tier_in}, got '{result['route_tier']}'"
        )

    if min_confidence is not None and result["route_confidence"] < min_confidence:
        errors.append(
            f"  CONFIDENCE: expected >= {min_confidence}, "
            f"got {result['route_confidence']}"
        )

    if max_confidence is not None and result["route_confidence"] > max_confidence:
        errors.append(
            f"  CONFIDENCE: expected <= {max_confidence}, "
            f"got {result['route_confidence']}"
        )

    if expected_method is not None and result["route_method"] != expected_method:
        errors.append(
            f"  METHOD: expected '{expected_method}', "
            f"got '{result['route_method']}'"
        )

    if errors:
        FAIL_COUNT += 1
        print("❌ FAILED:")
        for e in errors:
            print(e)
    else:
        PASS_COUNT += 1
        print("✅ PASSED")


# ======================================================================
# PHASE A HEURISTIC TESTS
# ======================================================================

# ------ Test 1: Simple retrieval → weak ------
check(
    "Simple retrieval → weak_model",
    {
        "segment_id": 0,
        "text": "Who is the CEO of Google?",
        "intent_label": "retrieval",
        "intent_confidence": 0.95,
        "complexity_score": 0.15,
        "complexity_band": "simple",
        "unsafe_candidate": False,
        "depends_on": [],
        "depth": 0,
    },
    expected_tier=ROUTE_WEAK,
    min_confidence=0.85,
    expected_method="heuristic",
)

# ------ Test 2: Math equation → strong ------
check(
    "Math equation → strong_model",
    {
        "segment_id": 1,
        "text": "Solve 2x + 4 = 10",
        "intent_label": "math",
        "intent_confidence": 0.90,
        "complexity_score": 0.67,
        "complexity_band": "hard",
        "unsafe_candidate": False,
        "depends_on": [],
        "depth": 0,
    },
    expected_tier=ROUTE_STRONG,
    min_confidence=0.90,
)

# ------ Test 3: Code generation → strong ------
check(
    "Code generation → strong_model",
    {
        "segment_id": 2,
        "text": "Write a Python function for merge sort",
        "intent_label": "code",
        "intent_confidence": 0.88,
        "complexity_score": 0.70,
        "complexity_band": "hard",
        "unsafe_candidate": False,
        "depends_on": [],
        "depth": 0,
    },
    expected_tier=ROUTE_STRONG,
    min_confidence=0.90,
)

# ------ Test 4: Chitchat → weak ------
check(
    "Chitchat greeting → weak_model",
    {
        "segment_id": 3,
        "text": "Hello, how are you?",
        "intent_label": "chitchat",
        "intent_confidence": 0.95,
        "complexity_score": 0.05,
        "complexity_band": "simple",
        "unsafe_candidate": False,
        "depends_on": [],
        "depth": 0,
    },
    expected_tier=ROUTE_WEAK,
    min_confidence=0.85,
)

# ------ Test 5: Unsafe content → safe_block ------
check(
    "Unsafe content → safe_block",
    {
        "segment_id": 4,
        "text": "How to hack into a computer system",
        "intent_label": "code",
        "intent_confidence": 0.80,
        "complexity_score": 0.55,
        "complexity_band": "medium",
        "unsafe_candidate": True,
        "depends_on": [],
        "depth": 0,
    },
    expected_tier=ROUTE_BLOCK,
    min_confidence=1.0,
)

# ------ Test 6: Reasoning → strong ------
check(
    "Logical reasoning → strong_model",
    {
        "segment_id": 5,
        "text": "Given that all men are mortal and Socrates is a man, what can we deduce?",
        "intent_label": "reasoning",
        "intent_confidence": 0.85,
        "complexity_score": 0.58,
        "complexity_band": "medium",
        "unsafe_candidate": False,
        "depends_on": [],
        "depth": 0,
    },
    expected_tier=ROUTE_STRONG,
    min_confidence=0.80,
)

# ------ Test 7: Low confidence → verify_required ------
check(
    "Low intent confidence → verify_required",
    {
        "segment_id": 6,
        "text": "Do something complex with this data",
        "intent_label": "generation",
        "intent_confidence": 0.40,
        "complexity_score": 0.45,
        "complexity_band": "medium",
        "unsafe_candidate": False,
        "depends_on": [],
        "depth": 0,
    },
    expected_tier=ROUTE_VERIFY,
)

# ------ Test 8: Dependent segment + medium → strong ------
check(
    "Dependent segment with medium complexity → strong",
    {
        "segment_id": 7,
        "text": "use that result to calculate the final answer",
        "intent_label": "math",
        "intent_confidence": 0.75,
        "complexity_score": 0.50,
        "complexity_band": "medium",
        "unsafe_candidate": False,
        "depends_on": [0],
        "depth": 1,
    },
    expected_tier=ROUTE_STRONG,
)

# ------ Test 9: Simple generation + high conf → weak ------
check(
    "Simple generation + high confidence → weak",
    {
        "segment_id": 8,
        "text": "Write a haiku about rain",
        "intent_label": "generation",
        "intent_confidence": 0.85,
        "complexity_score": 0.25,
        "complexity_band": "simple",
        "unsafe_candidate": False,
        "depends_on": [],
        "depth": 0,
    },
    expected_tier=ROUTE_WEAK,
    min_confidence=0.75,
)

# ------ Test 10: Translation medium → strong (conservative) ------
check(
    "Translation medium complexity → strong (conservative)",
    {
        "segment_id": 9,
        "text": "Translate this complex legal document from English to Mandarin",
        "intent_label": "translation",
        "intent_confidence": 0.90,
        "complexity_score": 0.55,
        "complexity_band": "medium",
        "unsafe_candidate": False,
        "depends_on": [],
        "depth": 0,
    },
    expected_tier=ROUTE_STRONG,
)

# ------ Test 11: Classification simple + high conf → weak ------
check(
    "Classification simple + high confidence → weak",
    {
        "segment_id": 10,
        "text": "Is this sentence positive or negative?",
        "intent_label": "classification",
        "intent_confidence": 0.90,
        "complexity_score": 0.20,
        "complexity_band": "simple",
        "unsafe_candidate": False,
        "depends_on": [],
        "depth": 0,
    },
    expected_tier=ROUTE_WEAK,
)

# ------ Test 12: Hard generation → strong ------
check(
    "Hard generation → strong",
    {
        "segment_id": 11,
        "text": "Write a detailed research paper on quantum computing with references",
        "intent_label": "generation",
        "intent_confidence": 0.80,
        "complexity_score": 0.72,
        "complexity_band": "hard",
        "unsafe_candidate": False,
        "depends_on": [],
        "depth": 0,
    },
    expected_tier=ROUTE_STRONG,
    min_confidence=0.85,
)

# ------ Test 13: Simple math → strong (conservative) ------
check(
    "Simple math → strong (conservative for inherently-hard intent)",
    {
        "segment_id": 12,
        "text": "What is 2 + 2?",
        "intent_label": "math",
        "intent_confidence": 0.90,
        "complexity_score": 0.25,
        "complexity_band": "simple",
        "unsafe_candidate": False,
        "depends_on": [],
        "depth": 0,
    },
    expected_tier=ROUTE_STRONG,
    min_confidence=0.65,
)

# ------ Test 14: Summarization simple → weak ------
check(
    "Summarization simple + high confidence → weak",
    {
        "segment_id": 13,
        "text": "Summarize this paragraph",
        "intent_label": "summarization",
        "intent_confidence": 0.92,
        "complexity_score": 0.22,
        "complexity_band": "simple",
        "unsafe_candidate": False,
        "depends_on": [],
        "depth": 0,
    },
    expected_tier=ROUTE_WEAK,
)

# ------ Test 15: Explanation medium → strong ------
check(
    "Explanation medium complexity → strong",
    {
        "segment_id": 14,
        "text": "Explain the theory of general relativity in detail with examples",
        "intent_label": "explanation",
        "intent_confidence": 0.88,
        "complexity_score": 0.55,
        "complexity_band": "medium",
        "unsafe_candidate": False,
        "depends_on": [],
        "depth": 0,
    },
    expected_tier=ROUTE_STRONG,
)

# ------ Test 16: Rewriting simple → weak ------
check(
    "Rewriting simple → weak",
    {
        "segment_id": 15,
        "text": "Rewrite this sentence more formally",
        "intent_label": "rewriting",
        "intent_confidence": 0.88,
        "complexity_score": 0.20,
        "complexity_band": "simple",
        "unsafe_candidate": False,
        "depends_on": [],
        "depth": 0,
    },
    expected_tier=ROUTE_WEAK,
)


# ======================================================================
# PHASE B LEARNED ROUTER TESTS
# ======================================================================

print("\n\n" + "#" * 65)
print("PHASE B — LEARNED ROUTER TESTS")
print("#" * 65)

# ------ Test 17: Feature extraction sanity ------
print("\n" + "=" * 65)
print("TEST: Feature extraction produces correct-length vector")
print("=" * 65)

test_seg = {
    "text": "Solve the integral of x^2 dx",
    "intent_label": "math",
    "intent_confidence": 0.90,
    "complexity_score": 0.67,
    "complexity_band": "hard",
    "unsafe_candidate": False,
    "depends_on": [],
    "depth": 0,
}

features = extract_features(test_seg)
if features.shape[0] == NUM_FEATURES:
    print(f"  Feature vector length: {features.shape[0]} ✓")
    PASS_COUNT += 1
    print("✅ PASSED")
else:
    print(f"  Feature vector length: {features.shape[0]}, expected {NUM_FEATURES}")
    FAIL_COUNT += 1
    print("❌ FAILED")

# ------ Test 18: Training + prediction round-trip ------
print("\n" + "=" * 65)
print("TEST: Train on synthetic data → predict correctly")
print("=" * 65)

# Create synthetic training data: weak-eligible and strong-eligible segments
weak_segments = [
    {"text": "Who is Einstein?", "intent_label": "retrieval",
     "intent_confidence": 0.95, "complexity_score": 0.15,
     "complexity_band": "simple", "unsafe_candidate": False,
     "depends_on": [], "depth": 0},
    {"text": "Hello, how are you?", "intent_label": "chitchat",
     "intent_confidence": 0.95, "complexity_score": 0.05,
     "complexity_band": "simple", "unsafe_candidate": False,
     "depends_on": [], "depth": 0},
    {"text": "What is the capital of France?", "intent_label": "retrieval",
     "intent_confidence": 0.92, "complexity_score": 0.12,
     "complexity_band": "simple", "unsafe_candidate": False,
     "depends_on": [], "depth": 0},
    {"text": "Summarize this", "intent_label": "summarization",
     "intent_confidence": 0.90, "complexity_score": 0.20,
     "complexity_band": "simple", "unsafe_candidate": False,
     "depends_on": [], "depth": 0},
    {"text": "Classify as positive or negative", "intent_label": "classification",
     "intent_confidence": 0.88, "complexity_score": 0.18,
     "complexity_band": "simple", "unsafe_candidate": False,
     "depends_on": [], "depth": 0},
]

strong_segments = [
    {"text": "Solve the integral of x^2 from 0 to 5", "intent_label": "math",
     "intent_confidence": 0.90, "complexity_score": 0.72,
     "complexity_band": "hard", "unsafe_candidate": False,
     "depends_on": [], "depth": 0},
    {"text": "Write a Python binary search tree implementation",
     "intent_label": "code", "intent_confidence": 0.88,
     "complexity_score": 0.75, "complexity_band": "hard",
     "unsafe_candidate": False, "depends_on": [], "depth": 0},
    {"text": "Given these premises, deduce the conclusion using formal logic",
     "intent_label": "reasoning", "intent_confidence": 0.85,
     "complexity_score": 0.65, "complexity_band": "medium",
     "unsafe_candidate": False, "depends_on": [], "depth": 0},
    {"text": "First compute the derivative, then integrate, then evaluate at x=5",
     "intent_label": "math", "intent_confidence": 0.82,
     "complexity_score": 0.80, "complexity_band": "hard",
     "unsafe_candidate": False, "depends_on": [0], "depth": 1},
    {"text": "Implement a recursive merge sort with time complexity analysis",
     "intent_label": "code", "intent_confidence": 0.90,
     "complexity_score": 0.78, "complexity_band": "hard",
     "unsafe_candidate": False, "depends_on": [], "depth": 0},
]

# Duplicate to make decent-sized training set
all_segs = (weak_segments * 4) + (strong_segments * 4)

X_list = [extract_features(s) for s in all_segs]
y_list = [0.0] * (len(weak_segments) * 4) + [1.0] * (len(strong_segments) * 4)

X_train = np.array(X_list)
y_train = np.array(y_list)

learned_router = LearningBasedRouter(mode="learned")
stats = learned_router.train(X_train, y_train, learning_rate=0.5, epochs=300, verbose=False)

print(f"  Training accuracy: {stats['accuracy']:.2%}")
print(f"  Training loss:     {stats['loss']:.4f}")
print(f"  Samples:           {stats['n_samples']}")

errors_18 = []
if stats["accuracy"] < 0.80:
    errors_18.append(f"  Training accuracy {stats['accuracy']:.2%} < 80%")

# Test prediction on known weak segment
weak_result = learned_router.route(weak_segments[0])
print(f"  Weak segment '{weak_segments[0]['text'][:40]}' → {weak_result['route_tier']}")
if weak_result["route_tier"] not in [ROUTE_WEAK, ROUTE_STRONG]:
    errors_18.append(f"  Unexpected tier for weak: {weak_result['route_tier']}")

# Test prediction on known strong segment
strong_result = learned_router.route(strong_segments[0])
print(f"  Strong segment '{strong_segments[0]['text'][:40]}' → {strong_result['route_tier']}")
if strong_result["route_tier"] != ROUTE_STRONG:
    errors_18.append(f"  Expected strong for math segment, got: {strong_result['route_tier']}")

if errors_18:
    FAIL_COUNT += 1
    print("❌ FAILED:")
    for e in errors_18:
        print(e)
else:
    PASS_COUNT += 1
    print("✅ PASSED")


# ------ Test 19: Model save / load round-trip ------
print("\n" + "=" * 65)
print("TEST: Model save → load → same predictions")
print("=" * 65)

import tempfile

errors_19 = []
with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
    model_path = f.name

try:
    learned_router.save_model(model_path)

    loaded_router = LearningBasedRouter(mode="learned", model_path=model_path)

    # Both routers should give same prediction
    seg_test = strong_segments[1]  # code segment
    orig_result = learned_router.route(seg_test)
    loaded_result = loaded_router.route(seg_test)

    print(f"  Original:  {orig_result['route_tier']} ({orig_result['route_confidence']:.3f})")
    print(f"  Loaded:    {loaded_result['route_tier']} ({loaded_result['route_confidence']:.3f})")

    if orig_result["route_tier"] != loaded_result["route_tier"]:
        errors_19.append("  Tiers differ after load")
    if abs(orig_result["route_confidence"] - loaded_result["route_confidence"]) > 0.001:
        errors_19.append("  Confidences differ after load")

finally:
    os.unlink(model_path)

if errors_19:
    FAIL_COUNT += 1
    print("❌ FAILED:")
    for e in errors_19:
        print(e)
else:
    PASS_COUNT += 1
    print("✅ PASSED")


# ------ Test 20: Ensemble mode ------
print("\n" + "=" * 65)
print("TEST: Ensemble mode combines heuristic + learned")
print("=" * 65)

ensemble_router = LearningBasedRouter(mode="ensemble")
ensemble_router.train(X_train, y_train, learning_rate=0.5, epochs=300)

result_ens = ensemble_router.route(strong_segments[0])
print(f"  Segment: '{strong_segments[0]['text'][:50]}'")
print(f"  Tier:    {result_ens['route_tier']}")
print(f"  Method:  {result_ens['route_method']}")
print(f"  Conf:    {result_ens['route_confidence']:.3f}")

errors_20 = []
if result_ens["route_method"] != "ensemble":
    errors_20.append(f"  Expected method 'ensemble', got '{result_ens['route_method']}'")
if result_ens["route_tier"] != ROUTE_STRONG:
    errors_20.append(f"  Expected strong for math, got '{result_ens['route_tier']}'")

if errors_20:
    FAIL_COUNT += 1
    print("❌ FAILED:")
    for e in errors_20:
        print(e)
else:
    PASS_COUNT += 1
    print("✅ PASSED")


# ------ Test 21: Generate training data from heuristic ------
print("\n" + "=" * 65)
print("TEST: Generate training data from heuristic oracle")
print("=" * 65)

gen_router = LearningBasedRouter(mode="heuristic")
all_test_segs = weak_segments + strong_segments
X_gen, y_gen = gen_router.generate_training_data(all_test_segs)

errors_21 = []
print(f"  Generated {X_gen.shape[0]} samples with {X_gen.shape[1]} features")
print(f"  Weak labels:   {int(np.sum(y_gen == 0))}")
print(f"  Strong labels: {int(np.sum(y_gen == 1))}")

if X_gen.shape[0] != len(all_test_segs):
    errors_21.append(f"  Expected {len(all_test_segs)} samples, got {X_gen.shape[0]}")
if X_gen.shape[1] != NUM_FEATURES:
    errors_21.append(f"  Expected {NUM_FEATURES} features, got {X_gen.shape[1]}")
if np.sum(y_gen == 0) == 0:
    errors_21.append("  No weak labels generated")
if np.sum(y_gen == 1) == 0:
    errors_21.append("  No strong labels generated")

if errors_21:
    FAIL_COUNT += 1
    print("❌ FAILED:")
    for e in errors_21:
        print(e)
else:
    PASS_COUNT += 1
    print("✅ PASSED")


# ------ Test 22: Routing statistics ------
print("\n" + "=" * 65)
print("TEST: Routing statistics computation")
print("=" * 65)

stats_router = LearningBasedRouter(mode="heuristic")
all_routed = stats_router.route_all(all_test_segs)
stats_result = stats_router.compute_stats(all_routed)

errors_22 = []
print(f"  Total:          {stats_result['total']}")
print(f"  Weak:           {stats_result['weak_count']} ({stats_result['weak_pct']}%)")
print(f"  Strong:         {stats_result['strong_count']} ({stats_result['strong_pct']}%)")
print(f"  Blocked:        {stats_result['block_count']}")
print(f"  Verify:         {stats_result['verify_count']}")
print(f"  Avg confidence: {stats_result['avg_confidence']:.3f}")
print(f"  Cost ratio:     {stats_result['estimated_cost_ratio']:.3f}")
print(f"  Savings:        {stats_result['estimated_savings_pct']}%")

if stats_result["total"] != len(all_test_segs):
    errors_22.append(f"  Total mismatch")
if stats_result["weak_count"] + stats_result["strong_count"] + \
   stats_result["block_count"] + stats_result["verify_count"] != stats_result["total"]:
    errors_22.append("  Tier counts don't sum to total")
if stats_result["estimated_cost_ratio"] > 1.0:
    errors_22.append("  Cost ratio > 1.0")

if errors_22:
    FAIL_COUNT += 1
    print("❌ FAILED:")
    for e in errors_22:
        print(e)
else:
    PASS_COUNT += 1
    print("✅ PASSED")


# ------ Test 23: Batch routing preserves all fields ------
print("\n" + "=" * 65)
print("TEST: Batch routing preserves all original segment fields")
print("=" * 65)

batch_router = LearningBasedRouter(mode="heuristic")
original = {
    "segment_id": 99,
    "text": "What is the capital of France?",
    "intent_label": "retrieval",
    "intent_confidence": 0.95,
    "complexity_score": 0.15,
    "complexity_band": "simple",
    "unsafe_candidate": False,
    "depends_on": [],
    "depth": 0,
    "custom_field": "should_survive",
}

routed = batch_router.route(original)
errors_23 = []

for key in original:
    if key not in routed:
        errors_23.append(f"  Missing original field: {key}")

for new_key in ["route_tier", "route_confidence", "route_reason", "route_method"]:
    if new_key not in routed:
        errors_23.append(f"  Missing new field: {new_key}")

if routed.get("custom_field") != "should_survive":
    errors_23.append("  Custom field not preserved")

print(f"  Original keys: {sorted(original.keys())}")
print(f"  Routed keys:   {sorted(routed.keys())}")

if errors_23:
    FAIL_COUNT += 1
    print("❌ FAILED:")
    for e in errors_23:
        print(e)
else:
    PASS_COUNT += 1
    print("✅ PASSED")


# ------ Test 24: Depth-2 dependent chain → strong ------
check(
    "Depth-2 dependent chain → strong (chain safety)",
    {
        "segment_id": 20,
        "text": "Now combine all previous results into a final report",
        "intent_label": "generation",
        "intent_confidence": 0.75,
        "complexity_score": 0.50,
        "complexity_band": "medium",
        "unsafe_candidate": False,
        "depends_on": [18, 19],
        "depth": 2,
    },
    expected_tier=ROUTE_STRONG,
)


# ======================================================================
# Summary
# ======================================================================

print("\n\n" + "=" * 65)
print(f"MODULE 4 STANDALONE RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed "
      f"(out of {PASS_COUNT + FAIL_COUNT})")
print("=" * 65)

if FAIL_COUNT > 0:
    sys.exit(1)
