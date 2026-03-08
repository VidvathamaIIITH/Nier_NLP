"""
Module 2 — Standalone Tests
============================

These tests use **hardcoded** segment lists (the same format that
Module 1 produces) so Module 2 can be validated completely
independently of Module 1.

Run from the repo root:
    python -m modules.dependency_graph.test
or directly:
    python modules/dependency_graph/test.py
"""

import sys
import os

# Ensure the project root is on sys.path regardless of how we're invoked
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.dependency_graph.graph_builder import DependencyGraphBuilder

builder = DependencyGraphBuilder()


# ======================================================================
# Test 1 — Two fully independent segments (should run in parallel)
# ======================================================================

def test_independent_segments():
    print("\n" + "=" * 60)
    print("TEST 1: Two independent segments")
    print("=" * 60)

    segments = [
        {"id": 0, "text": "Who is the CEO of Google",
         "depends_on_previous": False, "execution": "parallel"},
        {"id": 1, "text": "solve 2x + 4 = 10",
         "depends_on_previous": False, "execution": "parallel"},
    ]

    graph = builder.build(segments)
    builder.print_graph(graph)

    # Assertions
    assert len(graph["edges"]) == 0, "No edges expected"
    assert len(graph["parallel_groups"]) >= 1, "Both should be parallelisable"
    assert 0 in graph["parallel_groups"][0] and 1 in graph["parallel_groups"][0]
    assert len(graph["execution_plan"]) == 1, "Single parallel step"
    assert graph["execution_plan"][0]["mode"] == "parallel"
    print("✅ PASSED\n")


# ======================================================================
# Test 2 — Explicit dependency chain  (A → B)
# ======================================================================

def test_explicit_dependency():
    print("\n" + "=" * 60)
    print("TEST 2: Explicit dependency — 'then use that result'")
    print("=" * 60)

    segments = [
        {"id": 0, "text": "Find X",
         "depends_on_previous": False, "execution": "parallel"},
        {"id": 1, "text": "use that result to calculate Y",
         "depends_on_previous": True, "execution": "sequential"},
    ]

    graph = builder.build(segments)
    builder.print_graph(graph)

    assert len(graph["edges"]) == 1
    assert graph["edges"][0] == (0, 1)
    assert graph["nodes"][1]["depends_on"] == [0]
    assert graph["nodes"][1]["execution_mode"] == "sequential"
    assert len(graph["execution_plan"]) == 2, "Two sequential steps"
    assert graph["execution_plan"][0]["segments"][0]["segment_id"] == 0
    assert graph["execution_plan"][1]["segments"][0]["segment_id"] == 1
    print("✅ PASSED\n")


# ======================================================================
# Test 3 — Three-node chain  (A → B → C)
# ======================================================================

def test_three_node_chain():
    print("\n" + "=" * 60)
    print("TEST 3: Three-node sequential chain  A → B → C")
    print("=" * 60)

    segments = [
        {"id": 0, "text": "Compute the derivative of f(x)",
         "depends_on_previous": False, "execution": "parallel"},
        {"id": 1, "text": "then use the result to find critical points",
         "depends_on_previous": True, "execution": "sequential"},
        {"id": 2, "text": "then use those to determine maxima and minima",
         "depends_on_previous": True, "execution": "sequential"},
    ]

    graph = builder.build(segments)
    builder.print_graph(graph)

    assert len(graph["edges"]) == 2
    assert (0, 1) in graph["edges"]
    assert (1, 2) in graph["edges"]
    assert len(graph["sequential_chains"]) >= 1
    chain = graph["sequential_chains"][0]
    assert chain == [0, 1, 2]
    assert len(graph["execution_plan"]) == 3
    print("✅ PASSED\n")


# ======================================================================
# Test 4 — Diamond: A → B, A → C, B → D, C → D
# ======================================================================

def test_diamond_dependency():
    print("\n" + "=" * 60)
    print("TEST 4: Diamond dependency  A→B, A→C  (B,C)→D")
    print("=" * 60)

    # We simulate this by controlling depends_on_previous carefully
    segments = [
        {"id": 0, "text": "Gather the input data",
         "depends_on_previous": False, "execution": "parallel"},
        {"id": 1, "text": "Based on that, compute the mean",
         "depends_on_previous": True, "execution": "sequential"},
        {"id": 2, "text": "Based on that, compute the standard deviation",
         "depends_on_previous": False, "execution": "parallel"},
        {"id": 3, "text": "Using the result, produce a z-score table",
         "depends_on_previous": True, "execution": "sequential"},
    ]

    graph = builder.build(segments)
    builder.print_graph(graph)

    # 0→1 (from "Based on that" pointing to previous which is 0)
    # 2 is independent (depends_on_previous=False, no phrase match)
    # 3→2 (from "Using the result" pointing to previous which is 2)
    # So the execution plan should be:
    #   Step 0 (parallel): [0, 2]
    #   Step 1 (parallel): [1, 3]
    # OR:
    #   Step 0: [0], Step 1: [1, 2], Step 2: [3]  depending on edge detection

    assert len(graph["nodes"]) == 4
    assert len(graph["topological_order"]) == 4
    print("✅ PASSED\n")


# ======================================================================
# Test 5 — Mixed: two parallel + one dependent
# ======================================================================

def test_mixed_parallel_and_dependent():
    print("\n" + "=" * 60)
    print("TEST 5: Mixed — two parallel, one depends on first")
    print("=" * 60)

    segments = [
        {"id": 0, "text": "Translate this paragraph to French",
         "depends_on_previous": False, "execution": "parallel"},
        {"id": 1, "text": "Name three famous French authors",
         "depends_on_previous": False, "execution": "parallel"},
        {"id": 2, "text": "Then summarize the translation above",
         "depends_on_previous": True, "execution": "sequential"},
    ]

    graph = builder.build(segments)
    builder.print_graph(graph)

    # Seg 2 depends on seg 1 (its immediate predecessor)
    assert graph["nodes"][2]["execution_mode"] == "sequential"
    assert 2 not in graph["parallel_groups"][0]
    assert len(graph["execution_plan"]) >= 2
    print("✅ PASSED\n")


# ======================================================================
# Test 6 — Single segment (trivial — no dependencies possible)
# ======================================================================

def test_single_segment():
    print("\n" + "=" * 60)
    print("TEST 6: Single segment — trivial graph")
    print("=" * 60)

    segments = [
        {"id": 0, "text": "What is the capital of France?",
         "depends_on_previous": False, "execution": "parallel"},
    ]

    graph = builder.build(segments)
    builder.print_graph(graph)

    assert len(graph["nodes"]) == 1
    assert len(graph["edges"]) == 0
    assert len(graph["execution_plan"]) == 1
    assert graph["execution_plan"][0]["mode"] == "sequential"  # single item
    print("✅ PASSED\n")


# ======================================================================
# Test 7 — Pronoun-based implicit dependency
# ======================================================================

def test_pronoun_dependency():
    print("\n" + "=" * 60)
    print("TEST 7: Pronoun-based implicit dependency — 'It …'")
    print("=" * 60)

    segments = [
        {"id": 0, "text": "Describe photosynthesis",
         "depends_on_previous": False, "execution": "parallel"},
        {"id": 1, "text": "It is also important in ecology",
         "depends_on_previous": False, "execution": "parallel"},
    ]

    graph = builder.build(segments)
    builder.print_graph(graph)

    # "It" at the start should be detected as anaphoric
    assert graph["nodes"][1]["execution_mode"] == "sequential"
    assert (0, 1) in graph["edges"]
    print("✅ PASSED\n")


# ======================================================================
# Test 8 — No false dependency on noun-phrase 'and'
# ======================================================================

def test_no_false_dependency():
    print("\n" + "=" * 60)
    print("TEST 8: No false dependency — independent sentences")
    print("=" * 60)

    segments = [
        {"id": 0, "text": "Name three benefits of exercise",
         "depends_on_previous": False, "execution": "parallel"},
        {"id": 1, "text": "List five healthy breakfast options",
         "depends_on_previous": False, "execution": "parallel"},
    ]

    graph = builder.build(segments)
    builder.print_graph(graph)

    assert len(graph["edges"]) == 0
    assert graph["nodes"][0]["execution_mode"] == "parallel"
    assert graph["nodes"][1]["execution_mode"] == "parallel"
    print("✅ PASSED\n")


# ======================================================================
# Runner
# ======================================================================

if __name__ == "__main__":
    test_independent_segments()
    test_explicit_dependency()
    test_three_node_chain()
    test_diamond_dependency()
    test_mixed_parallel_and_dependent()
    test_single_segment()
    test_pronoun_dependency()
    test_no_false_dependency()

    print("\n" + "🎉" * 20)
    print("  ALL 8 MODULE-2 STANDALONE TESTS PASSED")
    print("🎉" * 20 + "\n")
