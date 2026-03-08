"""
Module 3 — Standalone Tests
============================

These tests use **hardcoded** segment dicts (same format Module 2 produces)
so Module 3 can be validated completely independently of Modules 1 & 2.

Run from the repo root:
    PYTHONPATH=. python modules/intent_complexity/test.py
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.intent_complexity.estimator import IntentComplexityEstimator

estimator = IntentComplexityEstimator()

PASS_COUNT = 0
FAIL_COUNT = 0


def check(test_name: str, segment: dict,
          expected_intent: str = None,
          expected_band: str = None,
          expect_unsafe: bool = None,
          min_complexity: float = None,
          max_complexity: float = None):
    """Run one test case and assert expectations."""
    global PASS_COUNT, FAIL_COUNT

    print("\n" + "=" * 65)
    print(f"TEST: {test_name}")
    print("=" * 65)

    result = estimator.estimate(segment)
    estimator.print_annotations([result], show_reasons=True)

    errors = []

    if expected_intent is not None and result["intent_label"] != expected_intent:
        errors.append(
            f"  INTENT: expected '{expected_intent}', got '{result['intent_label']}'"
        )

    if expected_band is not None and result["complexity_band"] != expected_band:
        errors.append(
            f"  BAND: expected '{expected_band}', got '{result['complexity_band']}'"
        )

    if expect_unsafe is not None and result["unsafe_candidate"] != expect_unsafe:
        errors.append(
            f"  UNSAFE: expected {expect_unsafe}, got {result['unsafe_candidate']}"
        )

    if min_complexity is not None and result["complexity_score"] < min_complexity:
        errors.append(
            f"  COMPLEXITY: expected >= {min_complexity}, got {result['complexity_score']}"
        )

    if max_complexity is not None and result["complexity_score"] > max_complexity:
        errors.append(
            f"  COMPLEXITY: expected <= {max_complexity}, got {result['complexity_score']}"
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
# Test 1 — Simple factual retrieval
# ======================================================================
check(
    "Simple factual retrieval",
    {"segment_id": 0, "text": "Who is the CEO of Google?",
     "depends_on": [], "depth": 0, "execution_mode": "parallel"},
    expected_intent="retrieval",
    expected_band="simple",
    expect_unsafe=False,
)

# ======================================================================
# Test 2 — Math problem
# ======================================================================
check(
    "Math equation",
    {"segment_id": 0, "text": "solve 2x + 4 = 10",
     "depends_on": [], "depth": 0, "execution_mode": "parallel"},
    expected_intent="math",
    min_complexity=0.5,
)

# ======================================================================
# Test 3 — Code task
# ======================================================================
check(
    "Code generation",
    {"segment_id": 0, "text": "Write a Python function to implement merge sort algorithm",
     "depends_on": [], "depth": 0, "execution_mode": "parallel"},
    expected_intent="code",
    expected_band="hard",
    expect_unsafe=False,
)

# ======================================================================
# Test 4 — Creative generation
# ======================================================================
check(
    "Creative generation — haiku",
    {"segment_id": 0, "text": "Write a haiku about rain",
     "depends_on": [], "depth": 0, "execution_mode": "parallel"},
    expected_intent="generation",
    max_complexity=0.5,
    expect_unsafe=False,
)

# ======================================================================
# Test 5 — Translation
# ======================================================================
check(
    "Translation request",
    {"segment_id": 0, "text": "Translate 'hello world' to French",
     "depends_on": [], "depth": 0, "execution_mode": "parallel"},
    expected_intent="translation",
    expect_unsafe=False,
)

# ======================================================================
# Test 6 — Classification
# ======================================================================
check(
    "Sentiment classification",
    {"segment_id": 0, "text": "Classify this review as positive or negative",
     "depends_on": [], "depth": 0, "execution_mode": "parallel"},
    expected_intent="classification",
    expected_band="simple",
    expect_unsafe=False,
)

# ======================================================================
# Test 7 — Chitchat
# ======================================================================
check(
    "Chitchat greeting",
    {"segment_id": 0, "text": "Hello, how are you?",
     "depends_on": [], "depth": 0, "execution_mode": "parallel"},
    expected_intent="chitchat",
    expected_band="simple",
    expect_unsafe=False,
)

# ======================================================================
# Test 8 — Summarization
# ======================================================================
check(
    "Summarization request",
    {"segment_id": 0, "text": "Summarize the main points of the article about climate change",
     "depends_on": [], "depth": 0, "execution_mode": "parallel"},
    expected_intent="summarization",
    expect_unsafe=False,
)

# ======================================================================
# Test 9 — Explanation
# ======================================================================
check(
    "Explanation request",
    {"segment_id": 0, "text": "Explain why photosynthesis is important for the ecosystem",
     "depends_on": [], "depth": 0, "execution_mode": "parallel"},
    expected_intent="explanation",
    expect_unsafe=False,
)

# ======================================================================
# Test 10 — Rewriting
# ======================================================================
check(
    "Rewriting request",
    {"segment_id": 0, "text": "Rewrite this paragraph to be more concise and formal",
     "depends_on": [], "depth": 0, "execution_mode": "parallel"},
    expected_intent="rewriting",
    expect_unsafe=False,
)

# ======================================================================
# Test 11 — Reasoning
# ======================================================================
check(
    "Logical reasoning",
    {"segment_id": 0,
     "text": "Given that all mammals are warm-blooded and whales are mammals, "
             "what can we deduce about whales? Justify your reasoning.",
     "depends_on": [], "depth": 0, "execution_mode": "parallel"},
    expected_intent="reasoning",
    min_complexity=0.4,
    expect_unsafe=False,
)

# ======================================================================
# Test 12 — Deep dependency → higher complexity
# ======================================================================
check(
    "Dependent segment at depth 2",
    {"segment_id": 2,
     "text": "use those results to determine the final answer",
     "depends_on": [1], "depth": 2, "execution_mode": "sequential"},
    min_complexity=0.4,
)

# ======================================================================
# Test 13 — Math word problem (longer)
# ======================================================================
check(
    "Math word problem",
    {"segment_id": 0,
     "text": "Amy bought a 15-foot spool of string. If she cuts up the "
             "entire string into an equal number of 6-inch and 12-inch "
             "wicks, what is the total number of wicks she will have cut?",
     "depends_on": [], "depth": 0, "execution_mode": "parallel"},
    expected_intent="math",
    expected_band="hard",
    expect_unsafe=False,
)

# ======================================================================
# Test 14 — Unsafe content flag
# ======================================================================
check(
    "Unsafe keyword detection",
    {"segment_id": 0,
     "text": "How to hack into someone's computer",
     "depends_on": [], "depth": 0, "execution_mode": "parallel"},
    expect_unsafe=True,
)

# ======================================================================
# Test 15 — Non-English text (translation signal)
# ======================================================================
check(
    "Non-English text",
    {"segment_id": 0,
     "text": "¿Qué diferencia al café cappuccino del café normal?",
     "depends_on": [], "depth": 0, "execution_mode": "parallel"},
    min_complexity=0.1,
    expect_unsafe=False,
)

# ======================================================================
# Test 16 — Multi-step reasoning with markers
# ======================================================================
check(
    "Multi-step reasoning with markers",
    {"segment_id": 0,
     "text": "First, identify the variables in the equation. "
             "Second, isolate x on one side. "
             "Third, solve for x. "
             "Finally, verify the solution by substituting back.",
     "depends_on": [], "depth": 0, "execution_mode": "parallel"},
    expected_band="hard",
)

# ======================================================================
# Summary
# ======================================================================
print("\n\n" + "=" * 65)
total = PASS_COUNT + FAIL_COUNT
print(f"  MODULE 3 STANDALONE: {PASS_COUNT}/{total} passed, {FAIL_COUNT} failed")
if FAIL_COUNT == 0:
    print("  🎉 ALL TESTS PASSED 🎉")
else:
    print("  ⚠  Some tests failed — review output above")
print("=" * 65 + "\n")

sys.exit(FAIL_COUNT)
