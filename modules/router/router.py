"""
Module 4 — Learning-Based Router
==================================

Converts intent / complexity annotations (Module 3) into concrete **routing
decisions** — which model tier should handle each segment.

Architecture (two-phase design from §3.2 "Select-Then-Route"):

  **Phase A — Heuristic Router (baseline):**
    A transparent, rule-based router that uses intent label, complexity band,
    dependency depth, unsafe flag, and confidence scores to decide routing.
    Every decision is fully explainable via the ``route_reasons`` list.

  **Phase B — Learned Router (research extension):**
    A lightweight scikit-learn classifier trained on segment-level features
    to predict ``P(needs_strong_model)``.  Falls back to heuristic when no
    trained model is available.

Routing targets:
  ``weak_model``      — cheap / fast tier (e.g. Llama-3-8B-Instruct)
  ``strong_model``    — expensive / capable tier (e.g. Llama-3-70B-Instruct)
  ``safe_block``      — unsafe content → block execution
  ``verify_required`` — route to strong + flag for post-hoc verification

Design philosophy:
  • Conservative errors are acceptable: over-routing to strong is fine.
  • False negatives are the most harmful: sending a hard problem to weak.
  • Every decision must carry a human-readable justification.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

ROUTE_WEAK = "weak_model"
ROUTE_STRONG = "strong_model"
ROUTE_BLOCK = "safe_block"
ROUTE_VERIFY = "verify_required"

# Intents that inherently need the strong model
STRONG_INTENTS = {"math", "code", "reasoning"}

# Intents that can typically be served by the weak model
WEAK_INTENTS = {"retrieval", "chitchat"}

# Intents that depend on complexity to decide
COMPLEXITY_DEPENDENT_INTENTS = {
    "generation", "translation", "classification",
    "summarization", "rewriting", "explanation",
}

# All valid intent labels (for one-hot encoding in Phase B)
ALL_INTENTS = [
    "retrieval", "reasoning", "math", "code", "generation",
    "translation", "classification", "summarization",
    "rewriting", "explanation", "chitchat",
]

# Feature names used by the learned router
LEARNED_FEATURE_NAMES = [
    "word_count",
    "char_count",
    "complexity_score",
    "intent_confidence",
    "depth",
    "is_dependent",
    "has_math_operator",
    "has_numbers",
    "has_code_fence",
    "has_question_mark",
    "math_signals",
    "code_signals",
    "reasoning_signals",
    "non_english_ratio",
    "unsafe_candidate",
    # 11 one-hot columns for intent labels
    *[f"intent_{lab}" for lab in ALL_INTENTS],
]

NUM_FEATURES = len(LEARNED_FEATURE_NAMES)


# ═══════════════════════════════════════════════════════════════════════
# Feature extraction for learned router
# ═══════════════════════════════════════════════════════════════════════

def extract_features(segment: Dict[str, Any]) -> np.ndarray:
    """
    Convert an annotated segment (Module 3 output) into a fixed-size
    numeric feature vector for the learned router.

    Returns
    -------
    np.ndarray  shape (NUM_FEATURES,)
    """
    text = segment.get("text", "")
    intent = segment.get("intent_label", "generation")

    # ── Scalar features ───────────────────────────────────────────
    word_count = len(text.split())
    char_count = len(text)
    complexity_score = segment.get("complexity_score", 0.3)
    intent_confidence = segment.get("intent_confidence", 0.5)
    depth = segment.get("depth", 0)
    is_dependent = 1.0 if segment.get("depends_on", []) else 0.0

    # Re-compute lightweight lexical features (they may not be stored)
    has_math_op = 1.0 if re.search(r"[=+\-*/^]", text) else 0.0
    has_numbers = 1.0 if re.search(r"\d", text) else 0.0
    has_code_fence = 1.0 if "```" in text else 0.0
    has_question = 1.0 if text.strip().endswith("?") else 0.0

    # Signal counts (quick re-scan; accuracy matches Module 3)
    math_signals = _quick_math_count(text)
    code_signals = _quick_code_count(text)
    reasoning_signals = _quick_reasoning_count(text)

    # Non-English ratio
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    non_ascii = sum(1 for ch in text if ord(ch) > 127 and not ch.isspace())
    non_english_ratio = non_ascii / alpha_chars if alpha_chars > 0 else 0.0

    unsafe_flag = 1.0 if segment.get("unsafe_candidate", False) else 0.0

    # ── One-hot intent encoding ───────────────────────────────────
    intent_one_hot = [1.0 if lab == intent else 0.0 for lab in ALL_INTENTS]

    vec = [
        word_count, char_count, complexity_score, intent_confidence,
        depth, is_dependent,
        has_math_op, has_numbers, has_code_fence, has_question,
        math_signals, code_signals, reasoning_signals,
        non_english_ratio, unsafe_flag,
        *intent_one_hot,
    ]
    return np.array(vec, dtype=np.float64)


# ── Quick signal counters (lightweight versions) ──────────────────────

_MATH_KW = {
    "solve", "calculate", "compute", "evaluate", "simplify",
    "integrate", "differentiate", "equation", "formula", "sum",
    "product", "percentage", "ratio", "average", "probability",
    "algebra", "geometry", "calculus", "matrix", "polynomial",
}

_CODE_KW = {
    "code", "program", "function", "class", "method", "algorithm",
    "implement", "debug", "compile", "script", "python", "java",
    "javascript", "typescript", "rust", "html", "sql", "api",
    "recursion", "binary", "stack", "queue", "sort", "search",
    "merge", "tree", "def ", "print(", "return ", "import ",
}

_REASONING_KW = {
    "reason", "reasoning", "logic", "logical", "argue", "analyse",
    "analyze", "evaluate", "infer", "deduce", "conclude",
    "therefore", "hence", "thus", "given that", "assuming",
    "if then", "evidence", "fallacy", "hypothesis",
}


def _quick_math_count(text: str) -> int:
    low = text.lower()
    score = sum(1 for kw in _MATH_KW if kw in low)
    score += len(re.findall(r"[=+\-*/^]", text))
    score += len(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    return score


def _quick_code_count(text: str) -> int:
    low = text.lower()
    return sum(1 for kw in _CODE_KW if kw in low)


def _quick_reasoning_count(text: str) -> int:
    low = text.lower()
    score = sum(1 for kw in _REASONING_KW if kw in low)
    for marker in ["first", "second", "third", "then", "finally",
                    "step 1", "step 2", "because", "since", "however"]:
        if marker in low:
            score += 1
    return score


# ═══════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════

class LearningBasedRouter:
    """
    Two-phase segment router.

    Phase A (always available):
        Deterministic heuristic rules based on intent, complexity,
        dependency, confidence, and safety flags.

    Phase B (optional, after training):
        A scikit-learn classifier that predicts P(needs_strong_model).
        Activated by calling ``train()`` or ``load_model()``.

    Usage::

        router = LearningBasedRouter()           # heuristic mode
        routed = router.route(annotated_segment)  # single segment
        routed_list = router.route_all(annotated_nodes)  # batch

        # --- Phase B ---
        router.train(X_features, y_labels)
        router.save_model("router_model.json")
        router.load_model("router_model.json")
    """

    # ── Configurable thresholds ────────────────────────────────────
    CONFIDENCE_UNCERTAIN_THRESHOLD = 0.55    # below → "uncertain"
    LEARNED_STRONG_THRESHOLD = 0.50          # P(strong) above → strong
    COMPLEXITY_WEAK_CEILING = 0.30           # max complexity for weak
    COMPLEXITY_STRONG_FLOOR = 0.55           # min complexity for force-strong

    def __init__(
        self,
        mode: str = "heuristic",
        model_path: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        mode : str
            ``"heuristic"``   — Phase A only (default)
            ``"learned"``     — Phase B if model available, else fallback
            ``"ensemble"``    — combine both (Phase B breaks ties)
        model_path : str, optional
            Path to a saved learned router model (JSON).
        """
        assert mode in {"heuristic", "learned", "ensemble"}, \
            f"Invalid mode '{mode}' — use 'heuristic', 'learned', or 'ensemble'"

        self.mode = mode

        # Phase B state
        self._learned_weights: Optional[np.ndarray] = None  # shape (NUM_FEATURES,)
        self._learned_bias: float = 0.0
        self._learned_ready: bool = False
        self._training_accuracy: float = 0.0
        self._training_samples: int = 0
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None

        if model_path and os.path.isfile(model_path):
            self.load_model(model_path)

    # ══════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════

    def route(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single annotated segment.

        Parameters
        ----------
        segment : dict
            Module 3 output — must have at least:
              text, intent_label, complexity_score, complexity_band,
              unsafe_candidate

        Returns
        -------
        dict
            Shallow copy of ``segment`` enriched with:
              route_tier       : str   — "weak_model" / "strong_model" /
                                         "safe_block" / "verify_required"
              route_confidence : float — 0.0–1.0
              route_reason     : str   — human-readable justification
              route_method     : str   — "heuristic" / "learned" / "ensemble"
        """
        # ── 0. Safety gate shortcut ───────────────────────────────
        if segment.get("unsafe_candidate", False):
            return self._emit(
                segment,
                tier=ROUTE_BLOCK,
                confidence=1.0,
                reason="Unsafe content detected — blocking execution",
                method="heuristic",
            )

        # ── 1. Phase A: heuristic decision ────────────────────────
        h_tier, h_conf, h_reason = self._heuristic_route(segment)

        # ── 2. Phase B: learned decision (if available) ───────────
        if self._learned_ready and self.mode in {"learned", "ensemble"}:
            l_tier, l_conf, l_reason = self._learned_route(segment)
        else:
            l_tier, l_conf, l_reason = None, 0.0, ""

        # ── 3. Combine / select ───────────────────────────────────
        if self.mode == "heuristic" or not self._learned_ready:
            return self._emit(segment, h_tier, h_conf, h_reason, "heuristic")

        elif self.mode == "learned" and self._learned_ready:
            return self._emit(segment, l_tier, l_conf, l_reason, "learned")

        else:  # ensemble
            tier, conf, reason = self._ensemble_decision(
                h_tier, h_conf, h_reason,
                l_tier, l_conf, l_reason,
            )
            return self._emit(segment, tier, conf, reason, "ensemble")

    def route_all(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Route a list of annotated segments (batch convenience)."""
        return [self.route(seg) for seg in segments]

    def route_graph(self, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Route all nodes in a Module-2 graph that have been annotated
        by Module 3.  Returns the enriched node list.
        """
        return self.route_all(graph.get("nodes", []))

    # ══════════════════════════════════════════════════════════════
    # Phase A — Heuristic Router
    # ══════════════════════════════════════════════════════════════

    def _heuristic_route(
        self, segment: Dict[str, Any]
    ) -> Tuple[str, float, str]:
        """
        Deterministic rule-based routing.

        Decision cascade:
          1. verify_required  — if intent confidence is very low
          2. strong_model     — if intent ∈ {math, code, reasoning}
          3. strong_model     — if complexity_band == "hard"
          4. strong_model     — if segment is dependent & complex
          5. weak_model       — if intent ∈ {retrieval, chitchat} & simple
          6. weak_model       — if complexity_band == "simple" & high confidence
          7. strong_model     — for complexity-dependent intents with medium+
          8. strong_model     — conservative fallback

        Returns (tier, confidence, reason)
        """
        intent = segment.get("intent_label", "generation")
        band = segment.get("complexity_band", "medium")
        score = segment.get("complexity_score", 0.5)
        intent_conf = segment.get("intent_confidence", 0.5)
        depth = segment.get("depth", 0)
        is_dep = bool(segment.get("depends_on", []))
        unsafe = segment.get("unsafe_candidate", False)

        reasons: List[str] = []

        # ── Rule 1: very low intent confidence → verify_required ──
        if intent_conf < self.CONFIDENCE_UNCERTAIN_THRESHOLD:
            reasons.append(
                f"Intent confidence {intent_conf:.2f} < "
                f"{self.CONFIDENCE_UNCERTAIN_THRESHOLD} (uncertain)"
            )
            reasons.append(
                f"Routing to strong with verification flag"
            )
            return ROUTE_VERIFY, 0.6, " | ".join(reasons)

        # ── Rule 2: inherently-strong intents ─────────────────────
        if intent in STRONG_INTENTS:
            reasons.append(f"Intent '{intent}' requires strong model")
            if band == "hard":
                reasons.append(f"Complexity band '{band}' confirms strong")
                return ROUTE_STRONG, 0.95, " | ".join(reasons)
            elif band == "medium":
                reasons.append(f"Complexity '{band}' — still strong for {intent}")
                return ROUTE_STRONG, 0.85, " | ".join(reasons)
            else:
                # Simple math/code — could be weak, but conservative
                reasons.append(
                    f"Complexity '{band}' but '{intent}' is inherently hard — "
                    f"routing strong (conservative)"
                )
                return ROUTE_STRONG, 0.70, " | ".join(reasons)

        # ── Rule 3: hard band → strong regardless of intent ──────
        if band == "hard":
            reasons.append(
                f"Complexity band 'hard' (score {score:.2f}) → strong model"
            )
            return ROUTE_STRONG, 0.90, " | ".join(reasons)

        # ── Rule 4: dependent + medium → strong (chain safety) ────
        if is_dep and band == "medium":
            reasons.append(
                f"Dependent segment (depth {depth}) with medium complexity "
                f"→ strong model (chain safety)"
            )
            return ROUTE_STRONG, 0.75, " | ".join(reasons)

        # ── Rule 5: weak-eligible intents + simple ────────────────
        if intent in WEAK_INTENTS and band == "simple":
            reasons.append(
                f"Intent '{intent}' with simple complexity → weak model"
            )
            return ROUTE_WEAK, 0.90, " | ".join(reasons)

        # ── Rule 6a: very low complexity → weak (trivially easy) ──
        if band == "simple" and score <= self.COMPLEXITY_WEAK_CEILING:
            reasons.append(
                f"Very low complexity (score {score:.2f} ≤ "
                f"{self.COMPLEXITY_WEAK_CEILING}) → weak model (trivially easy)"
            )
            return ROUTE_WEAK, 0.85, " | ".join(reasons)

        # ── Rule 6b: simple band + high confidence → weak ─────────
        if band == "simple" and intent_conf >= 0.70:
            reasons.append(
                f"Simple complexity (score {score:.2f}) with high "
                f"intent confidence ({intent_conf:.2f}) → weak model"
            )
            return ROUTE_WEAK, 0.80, " | ".join(reasons)

        # ── Rule 7: complexity-dependent intents with medium ──────
        if intent in COMPLEXITY_DEPENDENT_INTENTS and band == "medium":
            # Use complexity score to fine-tune
            if score >= self.COMPLEXITY_STRONG_FLOOR:
                reasons.append(
                    f"Intent '{intent}' with medium-high complexity "
                    f"({score:.2f}) → strong model"
                )
                return ROUTE_STRONG, 0.75, " | ".join(reasons)
            elif score <= self.COMPLEXITY_WEAK_CEILING:
                reasons.append(
                    f"Intent '{intent}' with medium-low complexity "
                    f"({score:.2f}) → weak model"
                )
                return ROUTE_WEAK, 0.65, " | ".join(reasons)
            else:
                # In the middle — go strong (conservative)
                reasons.append(
                    f"Intent '{intent}' with mid-range complexity "
                    f"({score:.2f}) → strong model (conservative)"
                )
                return ROUTE_STRONG, 0.60, " | ".join(reasons)

        # ── Rule 8: fallback → strong (conservative default) ──────
        reasons.append(
            f"No weak-eligible rule matched (intent='{intent}', "
            f"band='{band}', score={score:.2f}) → strong model (fallback)"
        )
        return ROUTE_STRONG, 0.55, " | ".join(reasons)

    # ══════════════════════════════════════════════════════════════
    # Phase B — Learned Router (logistic regression)
    # ══════════════════════════════════════════════════════════════

    def _learned_route(
        self, segment: Dict[str, Any]
    ) -> Tuple[str, float, str]:
        """
        Use the trained logistic regression model to predict P(strong).

        Returns (tier, confidence, reason).
        """
        if not self._learned_ready:
            return self._heuristic_route(segment)

        features = extract_features(segment)
        # Normalise using training statistics
        if self._feature_mean is not None and self._feature_std is not None:
            features = (features - self._feature_mean) / (self._feature_std + 1e-8)
        logit = float(np.dot(self._learned_weights, features) + self._learned_bias)
        p_strong = self._sigmoid(np.array([logit]))[0]   # numerically stable

        if p_strong >= self.LEARNED_STRONG_THRESHOLD:
            tier = ROUTE_STRONG
            confidence = p_strong
            reason = (
                f"Learned router: P(strong) = {p_strong:.3f} "
                f"≥ {self.LEARNED_STRONG_THRESHOLD} → strong model"
            )
        else:
            tier = ROUTE_WEAK
            confidence = 1.0 - p_strong
            reason = (
                f"Learned router: P(strong) = {p_strong:.3f} "
                f"< {self.LEARNED_STRONG_THRESHOLD} → weak model"
            )

        return tier, confidence, reason

    # ── Ensemble ──────────────────────────────────────────────────

    def _ensemble_decision(
        self,
        h_tier: str, h_conf: float, h_reason: str,
        l_tier: Optional[str], l_conf: float, l_reason: str,
    ) -> Tuple[str, float, str]:
        """
        Combine heuristic and learned decisions.

        Policy:
          • If both agree → use that with max(confidence).
          • If they disagree → use the more conservative one (strong),
            with averaged confidence, unless the learned model is very
            confident about weak.
        """
        if l_tier is None:
            return h_tier, h_conf, h_reason + " [learned unavailable]"

        if h_tier == l_tier:
            # Agreement
            combined_conf = max(h_conf, l_conf)
            reason = (
                f"Ensemble AGREE: {h_tier} | "
                f"heuristic({h_conf:.2f}), learned({l_conf:.2f})"
            )
            return h_tier, combined_conf, reason

        # Disagreement — be conservative
        # Only trust "weak" from the learned model if it's very confident
        if l_tier == ROUTE_WEAK and l_conf >= 0.80:
            reason = (
                f"Ensemble DISAGREE: heuristic→{h_tier}({h_conf:.2f}), "
                f"learned→{l_tier}({l_conf:.2f}) — trusting learned (high conf)"
            )
            return l_tier, l_conf * 0.9, reason

        # Otherwise, go conservative (strong)
        combined_conf = (h_conf + l_conf) / 2.0
        reason = (
            f"Ensemble DISAGREE: heuristic→{h_tier}({h_conf:.2f}), "
            f"learned→{l_tier}({l_conf:.2f}) — conservative → strong"
        )
        return ROUTE_STRONG, combined_conf, reason

    # ══════════════════════════════════════════════════════════════
    # Phase B — Training
    # ══════════════════════════════════════════════════════════════

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.1,
        epochs: int = 500,
        reg_lambda: float = 0.01,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Train a logistic regression classifier from scratch.

        Uses gradient descent with L2 regularisation — no sklearn needed
        so the module stays dependency-light.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, NUM_FEATURES)
            Feature matrix (use ``extract_features`` per sample).
        y : np.ndarray, shape (n_samples,)
            Binary labels: 1 = needs_strong, 0 = can_use_weak.
        learning_rate : float
            Step size for gradient descent.
        epochs : int
            Number of training iterations.
        reg_lambda : float
            L2 regularisation strength.
        verbose : bool
            Print loss every 50 epochs.

        Returns
        -------
        dict
            Training statistics: {accuracy, loss, epochs, n_samples}.
        """
        n, d = X.shape
        assert d == NUM_FEATURES, \
            f"Expected {NUM_FEATURES} features, got {d}"

        # Normalise features (store mean/std for inference)
        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0)
        X = (X - self._feature_mean) / (self._feature_std + 1e-8)

        # Initialise weights
        w = np.zeros(d)
        b = 0.0

        for epoch in range(epochs):
            # Forward pass
            logits = X @ w + b
            preds = self._sigmoid(logits)

            # Binary cross-entropy loss + L2
            eps = 1e-12
            loss = -np.mean(
                y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps)
            ) + 0.5 * reg_lambda * np.sum(w ** 2)

            # Gradients
            errors = preds - y
            dw = (X.T @ errors) / n + reg_lambda * w
            db = np.mean(errors)

            # Update
            w -= learning_rate * dw
            b -= learning_rate * db

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}  loss={loss:.4f}")

        # Store model
        self._learned_weights = w
        self._learned_bias = float(b)
        self._learned_ready = True

        # Compute training accuracy (X is already normalised above)
        final_preds = (self._sigmoid(X @ w + b) >= 0.5).astype(int)
        accuracy = float(np.mean(final_preds == y))
        self._training_accuracy = accuracy
        self._training_samples = n

        stats = {
            "accuracy": accuracy,
            "loss": float(loss),
            "epochs": epochs,
            "n_samples": n,
            "n_features": d,
        }
        return stats

    # ══════════════════════════════════════════════════════════════
    # Model persistence
    # ══════════════════════════════════════════════════════════════

    def save_model(self, path: str) -> None:
        """Persist the learned weights to a JSON file."""
        if not self._learned_ready:
            raise RuntimeError("No trained model to save.")

        payload = {
            "weights": self._learned_weights.tolist(),
            "bias": self._learned_bias,
            "n_features": NUM_FEATURES,
            "feature_names": LEARNED_FEATURE_NAMES,
            "training_accuracy": self._training_accuracy,
            "training_samples": self._training_samples,
            "strong_threshold": self.LEARNED_STRONG_THRESHOLD,
            "feature_mean": self._feature_mean.tolist() if self._feature_mean is not None else None,
            "feature_std": self._feature_std.tolist() if self._feature_std is not None else None,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load_model(self, path: str) -> None:
        """Load a previously saved learned router model."""
        with open(path, "r") as f:
            payload = json.load(f)

        assert payload["n_features"] == NUM_FEATURES, \
            f"Feature count mismatch: model has {payload['n_features']}, " \
            f"expected {NUM_FEATURES}"

        self._learned_weights = np.array(payload["weights"])
        self._learned_bias = float(payload["bias"])
        self._learned_ready = True
        self._training_accuracy = payload.get("training_accuracy", 0.0)
        self._training_samples = payload.get("training_samples", 0)
        if payload.get("feature_mean") is not None:
            self._feature_mean = np.array(payload["feature_mean"])
            self._feature_std = np.array(payload["feature_std"])

    # ══════════════════════════════════════════════════════════════
    # Synthetic training-data generator
    # ══════════════════════════════════════════════════════════════

    def generate_training_data(
        self, annotated_segments: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate (X, y) training data from annotated segments by using
        the **heuristic router as the oracle** (teacher).

        This bootstraps Phase B from Phase A's decisions, allowing
        the learned model to generalise beyond the explicit rules.

        Parameters
        ----------
        annotated_segments : list[dict]
            Module 3 outputs — segments with intent/complexity fields.

        Returns
        -------
        (X, y) : tuple[np.ndarray, np.ndarray]
            X has shape (n, NUM_FEATURES), y has shape (n,).
            y[i] = 1 if the heuristic routes to strong/verify, 0 if weak.
        """
        X_list = []
        y_list = []

        for seg in annotated_segments:
            # Skip blocked (unsafe) — those never reach routing
            if seg.get("unsafe_candidate", False):
                continue

            features = extract_features(seg)
            tier, _, _ = self._heuristic_route(seg)
            label = 1.0 if tier in {ROUTE_STRONG, ROUTE_VERIFY} else 0.0

            X_list.append(features)
            y_list.append(label)

        X = np.array(X_list, dtype=np.float64) if X_list else np.zeros((0, NUM_FEATURES))
        y = np.array(y_list, dtype=np.float64) if y_list else np.zeros(0)
        return X, y

    # ══════════════════════════════════════════════════════════════
    # Routing statistics
    # ══════════════════════════════════════════════════════════════

    def compute_stats(
        self, routed_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute summary statistics over a batch of routed segments.

        Returns
        -------
        dict with keys:
            total, weak_count, strong_count, block_count, verify_count,
            weak_pct, strong_pct, avg_confidence, estimated_cost_ratio
        """
        total = len(routed_segments)
        if total == 0:
            return {"total": 0}

        weak = sum(1 for s in routed_segments if s.get("route_tier") == ROUTE_WEAK)
        strong = sum(1 for s in routed_segments if s.get("route_tier") == ROUTE_STRONG)
        block = sum(1 for s in routed_segments if s.get("route_tier") == ROUTE_BLOCK)
        verify = sum(1 for s in routed_segments if s.get("route_tier") == ROUTE_VERIFY)

        avg_conf = sum(
            s.get("route_confidence", 0) for s in routed_segments
        ) / total

        # Cost ratio: if all went to strong = 1.0
        # weak ≈ 0.1x cost of strong  (based on Groq pricing)
        cost_factor = (weak * 0.1 + strong * 1.0 + verify * 1.0) / total
        savings_pct = (1.0 - cost_factor) * 100

        return {
            "total": total,
            "weak_count": weak,
            "strong_count": strong,
            "block_count": block,
            "verify_count": verify,
            "weak_pct": round(weak / total * 100, 1),
            "strong_pct": round(strong / total * 100, 1),
            "avg_confidence": round(avg_conf, 3),
            "estimated_cost_ratio": round(cost_factor, 3),
            "estimated_savings_pct": round(savings_pct, 1),
        }

    # ══════════════════════════════════════════════════════════════
    # Internals
    # ══════════════════════════════════════════════════════════════

    def _emit(
        self,
        segment: Dict[str, Any],
        tier: str,
        confidence: float,
        reason: str,
        method: str,
    ) -> Dict[str, Any]:
        """Build the enriched output dict."""
        result = dict(segment)   # shallow copy
        result.update({
            "route_tier": tier,
            "route_confidence": round(confidence, 3),
            "route_reason": reason,
            "route_method": method,
        })
        return result

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically-stable sigmoid."""
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z)),
        )

    # ══════════════════════════════════════════════════════════════
    # Pretty-print
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def print_routes(
        routed_segments: List[Dict[str, Any]],
        show_reason: bool = True,
    ) -> None:
        """Print a human-readable routing summary."""
        print("=" * 70)
        print("ROUTING DECISIONS")
        print("=" * 70)

        for seg in routed_segments:
            sid = seg.get("segment_id", seg.get("id", "?"))
            text = seg.get("text", "")
            tier = seg.get("route_tier", "?")
            conf = seg.get("route_confidence", 0)
            method = seg.get("route_method", "?")
            intent = seg.get("intent_label", "?")
            band = seg.get("complexity_band", "?")

            tier_icon = {
                ROUTE_WEAK: "🟢",
                ROUTE_STRONG: "🔴",
                ROUTE_BLOCK: "🚫",
                ROUTE_VERIFY: "⚠️",
            }.get(tier, "❓")

            print(f"\n  [{sid}] \"{text[:70]}{'…' if len(text) > 70 else ''}\"")
            print(f"       intent     = {intent} ({band})")
            print(f"       {tier_icon} route  = {tier} "
                  f"(confidence {conf:.2f}, via {method})")

            if show_reason:
                reason = seg.get("route_reason", "")
                print(f"       reason     = {reason}")

        print("\n" + "=" * 70)
