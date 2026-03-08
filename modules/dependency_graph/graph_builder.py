"""
Module 2 — Dependency Graph Builder
====================================

Takes the flat segment list produced by Module 1 (Semantic Decomposer) and
constructs a Directed Acyclic Graph (DAG) that encodes:

  • which segments are **independent** and can be executed in parallel
  • which segments **depend** on earlier ones and must run sequentially

The graph is exposed as:
  - ``nodes``   — enriched segment dicts
  - ``edges``   — list of (source_id → target_id) tuples
  - ``adjacency`` — forward adjacency list
  - ``parallel_groups`` — list of sets that may run concurrently
  - ``sequential_chains`` — ordered chains of dependent segments
  - ``topological_order`` — a valid execution schedule

Design goals (from the paper §3.1 "Dependency Filtering"):
  1. Rule-based detection of anaphoric / result-passing references.
  2. Conservative: if uncertain, mark sequential (safety over speed).
  3. Acyclic guarantee — topological sort must always succeed.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import spacy


# ---------------------------------------------------------------------------
# Dataclass-style helpers (plain dicts so the rest of the pipeline can
# serialise them to JSON without any special encoder).
# ---------------------------------------------------------------------------

def _make_node(segment: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a Module-1 segment into a graph node with extra metadata."""
    return {
        "segment_id": segment["id"],
        "text": segment["text"],
        "depends_on": [],           # list of segment_ids this node waits for
        "execution_mode": "parallel",
        "depth": 0,                 # distance from a root node
    }


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class DependencyGraphBuilder:
    """Build and query a segment-level dependency DAG."""

    # ---- Lexical patterns that signal a segment depends on a prior one ------

    # Explicit backward-reference phrases
    DEPENDENCY_PHRASES: List[str] = [
        "use that",
        "use it",
        "use the result",
        "use the answer",
        "use the above",
        "use the output",
        "use the value",
        "use those",
        "use these",
        "using that",
        "using it",
        "using the result",
        "using the answer",
        "using the above",
        "using those",
        "with that",
        "with the result",
        "with the answer",
        "with it",
        "based on that",
        "based on the result",
        "based on the answer",
        "based on the above",
        "based on it",
        "from the result",
        "from the answer",
        "from that",
        "from it",
        "that result",
        "the result",
        "the answer",
        "the output",
        "the value",
        "then use",
        "then calculate",
        "then compute",
        "then solve",
        "then find",
        "then output",
        "then return",
        "then explain",
        "then summarize",
        "then generate",
        "then write",
        "then translate",
        "then list",
        "then compare",
        "then describe",
        "after that",
        "after computing",
        "after solving",
        "after finding",
        "once you have",
        "once you get",
        "once computed",
        "now use",
        "now calculate",
        "now compute",
        "now solve",
        "now find",
        "now output",
        "plug that",
        "plug it",
        "substitute that",
        "substitute it",
        "apply that",
        "apply it",
        "take that",
        "take the result",
    ]

    # Pronouns / anaphora that point backwards
    ANAPHORIC_STARTS: Set[str] = {
        "it", "its",
        "they", "them", "their",
        "that", "those", "these", "this",
        "he", "she",
        "the former", "the latter",
    }

    # Temporal / sequential discourse markers at segment start
    SEQUENTIAL_MARKERS: Set[str] = {
        "then", "next", "finally", "afterwards",
        "subsequently", "after", "once",
        "now", "lastly", "second", "third",
    }

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Optionally load spaCy for pronoun-resolution heuristics."""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")

    # ------------------------------------------------------------------
    # Dependency detection helpers
    # ------------------------------------------------------------------

    def _has_dependency_phrase(self, text: str) -> bool:
        """Check whether *text* contains an explicit backward-reference."""
        lowered = text.lower()
        return any(phrase in lowered for phrase in self.DEPENDENCY_PHRASES)

    def _starts_with_anaphora(self, text: str) -> bool:
        """Check if the segment starts with an anaphoric pronoun."""
        lowered = text.lower().strip()
        for marker in self.ANAPHORIC_STARTS:
            if lowered.startswith(marker) and (
                len(lowered) == len(marker)
                or not lowered[len(marker)].isalpha()
            ):
                return True
        return False

    def _starts_with_sequential_marker(self, text: str) -> bool:
        """Check if the segment opens with a temporal sequencing word."""
        lowered = text.lower().strip()
        for marker in self.SEQUENTIAL_MARKERS:
            if lowered.startswith(marker) and (
                len(lowered) == len(marker)
                or not lowered[len(marker)].isalpha()
            ):
                return True
        return False

    def _has_pronoun_reference(self, text: str) -> bool:
        """Use spaCy POS tagging to find a pronoun in subject position."""
        doc = self.nlp(text)
        for token in doc:
            if (
                token.pos_ == "PRON"
                and token.dep_ in {"nsubj", "nsubjpass", "dobj", "pobj"}
                and token.lower_ in {
                    "it", "its", "they", "them", "their",
                    "that", "those", "these", "this",
                }
            ):
                return True
        return False

    def _detect_dependency(
        self,
        segment: Dict[str, Any],
        previous_segments: List[Dict[str, Any]],
    ) -> Optional[int]:
        """
        Determine whether *segment* depends on any earlier segment.

        Returns the **segment_id** it depends on, or ``None``.

        Resolution order (first match wins):
          1. Module-1 already flagged ``depends_on_previous=True``.
          2. Explicit dependency phrase detected.
          3. Segment starts with a sequential discourse marker.
          4. Segment starts with an anaphoric pronoun.
          5. spaCy finds a pronoun in a dependency-relevant position.
        """
        if not previous_segments:
            return None

        text = segment["text"]

        # Respect Module-1's own flag
        if segment.get("depends_on_previous", False):
            return previous_segments[-1]["segment_id"]

        if self._has_dependency_phrase(text):
            return previous_segments[-1]["segment_id"]

        if self._starts_with_sequential_marker(text):
            return previous_segments[-1]["segment_id"]

        if self._starts_with_anaphora(text):
            return previous_segments[-1]["segment_id"]

        if self._has_pronoun_reference(text):
            return previous_segments[-1]["segment_id"]

        return None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build(
        self,
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build the full dependency graph from a Module-1 segment list.

        Parameters
        ----------
        segments : list[dict]
            Output of ``SemanticDecomposer.decompose()``.
            Each dict has at least ``id``, ``text``,
            ``depends_on_previous``, ``execution``.

        Returns
        -------
        dict  with keys:
            nodes              — list of enriched node dicts
            edges              — list of (src, dst) tuples
            adjacency          — dict[int, list[int]]
            reverse_adjacency  — dict[int, list[int]]
            parallel_groups    — list of sets of segment_ids
            sequential_chains  — list of ordered id-lists
            topological_order  — list of segment_ids
            execution_plan     — list of steps, each a dict
                                 {"step": int, "segments": [...],
                                  "mode": "parallel"|"sequential"}
        """

        # -- 1. Create nodes -----------------------------------------------
        nodes: List[Dict[str, Any]] = []
        node_map: Dict[int, Dict[str, Any]] = {}

        for seg in segments:
            node = _make_node(seg)
            nodes.append(node)
            node_map[node["segment_id"]] = node

        # -- 2. Detect edges -----------------------------------------------
        edges: List[Tuple[int, int]] = []
        adjacency: Dict[int, List[int]] = defaultdict(list)
        reverse_adj: Dict[int, List[int]] = defaultdict(list)

        for idx, node in enumerate(nodes):
            preceding = nodes[:idx]
            dep_target = self._detect_dependency(
                segment=segments[idx],
                previous_segments=preceding,
            )
            if dep_target is not None:
                node["depends_on"].append(dep_target)
                node["execution_mode"] = "sequential"
                edges.append((dep_target, node["segment_id"]))
                adjacency[dep_target].append(node["segment_id"])
                reverse_adj[node["segment_id"]].append(dep_target)

        # -- 3. Compute depth for every node --------------------------------
        for node in nodes:
            node["depth"] = self._compute_depth(
                node["segment_id"], reverse_adj
            )

        # -- 4. Topological sort --------------------------------------------
        topo_order = self._topological_sort(nodes, adjacency)

        # -- 5. Extract parallel groups & sequential chains -----------------
        parallel_groups = self._extract_parallel_groups(nodes, edges)
        sequential_chains = self._extract_sequential_chains(nodes, adjacency)

        # -- 6. Build a human-readable execution plan -----------------------
        execution_plan = self._build_execution_plan(
            topo_order, node_map, adjacency, reverse_adj
        )

        return {
            "nodes": nodes,
            "edges": edges,
            "adjacency": dict(adjacency),
            "reverse_adjacency": dict(reverse_adj),
            "parallel_groups": parallel_groups,
            "sequential_chains": sequential_chains,
            "topological_order": topo_order,
            "execution_plan": execution_plan,
        }

    # ------------------------------------------------------------------
    # Internal graph algorithms
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_depth(
        node_id: int,
        reverse_adj: Dict[int, List[int]],
    ) -> int:
        """BFS backwards to find the longest path from a root."""
        visited: Set[int] = set()
        depth = 0
        frontier = [node_id]
        while frontier:
            next_frontier = []
            for nid in frontier:
                for parent in reverse_adj.get(nid, []):
                    if parent not in visited:
                        visited.add(parent)
                        next_frontier.append(parent)
            if next_frontier:
                depth += 1
            frontier = next_frontier
        return depth

    @staticmethod
    def _topological_sort(
        nodes: List[Dict[str, Any]],
        adjacency: Dict[int, List[int]],
    ) -> List[int]:
        """Kahn's algorithm — returns a valid execution order."""
        in_degree: Dict[int, int] = {n["segment_id"]: 0 for n in nodes}
        for src, dsts in adjacency.items():
            for dst in dsts:
                in_degree[dst] = in_degree.get(dst, 0) + 1

        queue: deque[int] = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        order: List[int] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for child in adjacency.get(nid, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(nodes):
            raise ValueError(
                "Cycle detected in dependency graph — this should never "
                "happen with well-formed segment lists."
            )
        return order

    @staticmethod
    def _extract_parallel_groups(
        nodes: List[Dict[str, Any]],
        edges: List[Tuple[int, int]],
    ) -> List[Set[int]]:
        """
        Group nodes by depth level; nodes at the same depth with no
        mutual edges may execute in parallel.
        """
        depth_buckets: Dict[int, Set[int]] = defaultdict(set)
        for node in nodes:
            depth_buckets[node["depth"]].add(node["segment_id"])

        edge_set = set(edges)
        groups: List[Set[int]] = []

        for depth in sorted(depth_buckets):
            bucket = depth_buckets[depth]
            # Filter out any pair that has a direct edge (shouldn't happen
            # at the same depth, but be safe).
            safe_group: Set[int] = set()
            for nid in bucket:
                conflict = False
                for other in safe_group:
                    if (nid, other) in edge_set or (other, nid) in edge_set:
                        conflict = True
                        break
                if not conflict:
                    safe_group.add(nid)
            if safe_group:
                groups.append(safe_group)
        return groups

    @staticmethod
    def _extract_sequential_chains(
        nodes: List[Dict[str, Any]],
        adjacency: Dict[int, List[int]],
    ) -> List[List[int]]:
        """
        Find maximal chains A→B→C where each node has exactly one
        child and each child has exactly one parent.
        """
        child_count: Dict[int, int] = {n["segment_id"]: 0 for n in nodes}
        parent_count: Dict[int, int] = {n["segment_id"]: 0 for n in nodes}
        for src, dsts in adjacency.items():
            child_count[src] = len(dsts)
            for dst in dsts:
                parent_count[dst] = parent_count.get(dst, 0) + 1

        visited: Set[int] = set()
        chains: List[List[int]] = []

        # Start chains from nodes with no parent or whose parent has
        # multiple children (i.e. chain heads).
        for node in nodes:
            nid = node["segment_id"]
            if nid in visited:
                continue
            if parent_count[nid] == 0 or parent_count[nid] > 1:
                # Try to walk a chain forward
                chain = [nid]
                visited.add(nid)
                current = nid
                while (
                    child_count.get(current, 0) == 1
                    and parent_count.get(adjacency[current][0], 0) == 1
                ):
                    nxt = adjacency[current][0]
                    if nxt in visited:
                        break
                    chain.append(nxt)
                    visited.add(nxt)
                    current = nxt
                if len(chain) > 1:
                    chains.append(chain)
        return chains

    @staticmethod
    def _build_execution_plan(
        topo_order: List[int],
        node_map: Dict[int, Dict[str, Any]],
        adjacency: Dict[int, List[int]],
        reverse_adj: Dict[int, List[int]],
    ) -> List[Dict[str, Any]]:
        """
        Convert topological order into concrete execution steps.

        Each step contains a batch of segments that can run at the same
        time.  Steps are ordered so that every dependency is resolved
        before the dependent step begins.
        """
        completed: Set[int] = set()
        remaining = list(topo_order)
        plan: List[Dict[str, Any]] = []
        step_num = 0

        while remaining:
            # Find all nodes whose parents are already completed
            ready = [
                nid for nid in remaining
                if all(p in completed for p in reverse_adj.get(nid, []))
            ]
            if not ready:
                # Should never happen after a valid topo-sort
                raise RuntimeError("Deadlock in execution plan construction")

            mode = "parallel" if len(ready) > 1 else "sequential"
            step_segments = []
            for nid in ready:
                step_segments.append({
                    "segment_id": nid,
                    "text": node_map[nid]["text"],
                    "depends_on": node_map[nid]["depends_on"],
                })

            plan.append({
                "step": step_num,
                "mode": mode,
                "segments": step_segments,
            })

            for nid in ready:
                completed.add(nid)
                remaining.remove(nid)
            step_num += 1

        return plan

    # ------------------------------------------------------------------
    # Pretty-print helpers
    # ------------------------------------------------------------------

    @staticmethod
    def print_graph(graph: Dict[str, Any]) -> None:
        """Human-readable summary of the dependency graph."""
        print("=" * 65)
        print("DEPENDENCY GRAPH SUMMARY")
        print("=" * 65)

        print(f"\nTotal segments : {len(graph['nodes'])}")
        print(f"Total edges    : {len(graph['edges'])}")

        # Nodes
        print("\n--- Nodes ---")
        for node in graph["nodes"]:
            dep_str = (
                f"depends_on={node['depends_on']}"
                if node["depends_on"]
                else "independent"
            )
            print(
                f"  [{node['segment_id']}] "
                f"depth={node['depth']}  "
                f"{dep_str}  "
                f"mode={node['execution_mode']}"
            )
            print(f"       \"{node['text'][:80]}{'…' if len(node['text']) > 80 else ''}\"")

        # Edges
        if graph["edges"]:
            print("\n--- Edges ---")
            for src, dst in graph["edges"]:
                print(f"  {src} ──▶ {dst}")

        # Parallel groups
        if graph["parallel_groups"]:
            print("\n--- Parallel Groups ---")
            for i, group in enumerate(graph["parallel_groups"]):
                print(f"  Group {i}: {sorted(group)}")

        # Sequential chains
        if graph["sequential_chains"]:
            print("\n--- Sequential Chains ---")
            for i, chain in enumerate(graph["sequential_chains"]):
                print(f"  Chain {i}: {' → '.join(str(c) for c in chain)}")

        # Execution plan
        print("\n--- Execution Plan ---")
        for step in graph["execution_plan"]:
            ids = [s["segment_id"] for s in step["segments"]]
            print(f"  Step {step['step']} ({step['mode']}): segments {ids}")

        print("=" * 65)
