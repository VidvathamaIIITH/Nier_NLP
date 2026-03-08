"""
Module 1 → Module 2  Integration Test
=======================================

Pipes real prompts through Module 1 (Semantic Decomposer) and feeds
the segment list directly into Module 2 (Dependency Graph Builder).

Run from the repo root:
    python -m modules.dependency_graph.test_integration
or directly:
    python modules/dependency_graph/test_integration.py
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.semantic_decomposition.semantic_decomposer import SemanticDecomposer
from modules.dependency_graph.graph_builder import DependencyGraphBuilder


decomposer = SemanticDecomposer()
graph_builder = DependencyGraphBuilder()


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def run_pipeline(label: str, prompt: str, expected_segments: int = None,
                 expected_edges: int = None, min_parallel: int = None):
    """Decompose → Build Graph → Print → Assert."""
    print("\n" + "=" * 65)
    print(f"INTEGRATION TEST: {label}")
    print("=" * 65)
    print(f"  PROMPT: {prompt[:100]}{'…' if len(prompt) > 100 else ''}\n")

    # --- Module 1 ---
    segments = decomposer.decompose(prompt)
    print("  Module 1 output:")
    for seg in segments:
        dep = "DEP" if seg.get("depends_on_previous") else "IND"
        print(f"    [{seg['id']}] ({dep}) {seg['text'][:70]}")

    # --- Module 2 ---
    graph = graph_builder.build(segments)
    print()
    graph_builder.print_graph(graph)

    # --- Assertions ---
    if expected_segments is not None:
        assert len(graph["nodes"]) == expected_segments, (
            f"Expected {expected_segments} segments, got {len(graph['nodes'])}"
        )
    if expected_edges is not None:
        assert len(graph["edges"]) == expected_edges, (
            f"Expected {expected_edges} edges, got {len(graph['edges'])}"
        )
    if min_parallel is not None:
        max_group = max(len(g) for g in graph["parallel_groups"]) if graph["parallel_groups"] else 0
        assert max_group >= min_parallel, (
            f"Expected at least {min_parallel} parallel segments in a group, "
            f"got {max_group}"
        )

    print("✅ PASSED\n")
    return graph


# ======================================================================
# Test Cases
# ======================================================================

# Case 1: Classic mixed-intent prompt — should split and be parallel
run_pipeline(
    label="Mixed-intent parallel",
    prompt="Who is the CEO of Google and solve 2x + 4 = 10",
    expected_segments=2,
    expected_edges=0,
    min_parallel=2,
)

# Case 2: Dependent chain — "Find X, then use that result"
run_pipeline(
    label="Dependent chain",
    prompt="Find X, then use that result to calculate Y.",
    expected_segments=2,
    expected_edges=1,
)

# Case 3: Single-task prompt — no split, no edges
run_pipeline(
    label="Single task (no split)",
    prompt="What is the capital of France?",
    expected_segments=1,
    expected_edges=0,
)

# Case 4: Noun-phrase 'and' should NOT split
run_pipeline(
    label="Noun-phrase 'and' (no false split)",
    prompt="Name a famous person who embodies the following values: knowledge and creativity.",
    expected_segments=1,
    expected_edges=0,
)

# Case 5: "Compare and contrast" should NOT split
run_pipeline(
    label="Compare and contrast (no false split)",
    prompt="Compare and contrast two popular tourist attractions in your hometown.",
    expected_segments=1,
    expected_edges=0,
)

# Case 6: Multi-sentence with one dependent
run_pipeline(
    label="Multi-sentence with dependent pronoun",
    prompt="Describe photosynthesis. Then explain why it matters for global ecology.",
    expected_edges=1,
)

# Case 7: Three independent instructions
run_pipeline(
    label="Three independent instructions",
    prompt="Translate 'hello' to French. Name three prime numbers. Write a haiku about rain.",
    expected_segments=3,
    expected_edges=0,
    min_parallel=3,
)

# Case 8: Long multi-step math problem — should stay as one segment
run_pipeline(
    label="Single math word problem (no split)",
    prompt=(
        "Amy bought a 15-foot spool of string to cut up into wicks for "
        "making candles. If she cuts up the entire string into an equal "
        "number of 6-inch and 12-inch wicks, what is the total number of "
        "wicks she will have cut?"
    ),
)

# Case 9: Code-block prompt — bypass splitting entirely
run_pipeline(
    label="Code block passthrough",
    prompt='Here is some code: ```python\nprint("hello")\n``` Explain what it does.',
    expected_segments=1,
    expected_edges=0,
)


# ======================================================================
# Summary
# ======================================================================
print("\n" + "🎉" * 20)
print("  ALL INTEGRATION TESTS PASSED — Module 1 → Module 2 pipeline OK")
print("🎉" * 20 + "\n")
