"""
Module 3 — Intent & Complexity Estimator
==========================================

Classifies each segment's **intent** (what kind of task it is) and estimates
its **complexity** (how hard it is).  This is the intelligence layer between
dependency analysis (Module 2) and routing (Module 4).

Design philosophy (from the paper §3.2):
  • Start with interpretable heuristics — keyword/regex/POS rules.
  • Every classification must be *explainable*: the ``reasons`` list
    records why a particular intent and complexity were chosen.
  • Complexity is a continuous score [0, 1] that also maps to a band
    (``simple``, ``medium``, ``hard``).

Intent taxonomy:
  retrieval, reasoning, math, code, generation, translation,
  classification, summarization, rewriting, explanation, chitchat

Flags (orthogonal to intent):
  unsafe_candidate — set when the text triggers a safety-relevant keyword
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════
# Keyword / pattern banks
# ═══════════════════════════════════════════════════════════════════════

# ── Intent-detection patterns ──────────────────────────────────────────

MATH_KEYWORDS: Set[str] = {
    "solve", "calculate", "compute", "evaluate", "simplify",
    "integrate", "differentiate", "derive", "prove", "factor",
    "equation", "inequality", "formula", "sum", "product",
    "percentage", "ratio", "proportion", "average", "mean",
    "median", "probability", "permutation", "combination",
    "algebra", "geometry", "trigonometry", "calculus",
    "arithmetic", "denominator", "numerator", "fraction",
    "exponent", "logarithm", "matrix", "determinant",
    "polynomial", "quadratic", "linear", "coefficient",
}

MATH_PATTERNS: List[re.Pattern] = [
    re.compile(r"[0-9]+\s*[+\-*/^=<>≤≥]\s*[0-9xXyY]", re.IGNORECASE),
    re.compile(r"\b\d+\s*(?:feet|foot|inch|cm|km|miles?|meters?|kg|lbs?|dollars?|\$|€|£|percent|%)\b", re.IGNORECASE),
    re.compile(r"\bx\s*[+\-*/^=]\s*\d", re.IGNORECASE),
    re.compile(r"\b\d+\s*/\s*\d+\b"),
    re.compile(r"\bhow\s+(?:many|much|far|long|old|fast|tall|heavy)\b", re.IGNORECASE),
]

CODE_KEYWORDS: Set[str] = {
    "code", "program", "function", "class", "method", "algorithm",
    "implement", "debug", "compile", "execute", "script",
    "python", "java", "javascript", "typescript", "c++", "rust",
    "html", "css", "sql", "api", "json", "xml", "yaml",
    "variable", "loop", "array", "list", "dictionary", "hash",
    "recursion", "iteration", "binary", "stack", "queue",
    "sort", "search", "merge", "linked list", "tree", "graph",
    "regex", "import", "module", "library", "framework",
    "def ", "print(", "return ", "if __name__",
    "console.log", "system.out", "public static",
}

CODE_PATTERNS: List[re.Pattern] = [
    re.compile(r"```\w*"),                           # fenced code block
    re.compile(r"\bdef\s+\w+\s*\("),                 # python function def
    re.compile(r"\bclass\s+\w+[\s(:]"),              # class definition
    re.compile(r"(?:for|while)\s*\("),               # C-style loop
    re.compile(r"(?:let|const|var)\s+\w+\s*="),      # JS variable
    re.compile(r"import\s+\w+"),                     # import statement
    re.compile(r"\bprint\s*\("),                     # print call
]

RETRIEVAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"^(?:who|what|when|where|which)\s+(?:is|are|was|were|did|does|do|has|have|had)\b", re.IGNORECASE),
    re.compile(r"^(?:tell me|give me|name)\s+(?:the|a|an)?\s*\w+", re.IGNORECASE),
    re.compile(r"\b(?:capital|president|ceo|founder|inventor|author|creator)\s+of\b", re.IGNORECASE),
    re.compile(r"\b(?:define|definition of|what does .+ mean)\b", re.IGNORECASE),
    re.compile(r"\b(?:how tall|how old|how many people|population of|area of)\b", re.IGNORECASE),
]

RETRIEVAL_KEYWORDS: Set[str] = {
    "who is", "what is", "when was", "where is", "which is",
    "tell me", "name the", "give me", "define",
    "capital of", "president of", "ceo of", "founder of",
}

GENERATION_KEYWORDS: Set[str] = {
    "write", "compose", "draft", "create", "generate",
    "come up with", "brainstorm", "suggest", "produce",
    "story", "poem", "essay", "article", "letter",
    "email", "blog", "post", "tweet", "haiku",
    "paragraph", "sentence", "lyrics", "speech",
    "dialogue", "script", "recipe", "outline",
    "design", "propose", "invent", "imagine",
}

TRANSLATION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\btranslat(?:e|ion)\b", re.IGNORECASE),
    re.compile(r"\b(?:in|to|into)\s+(?:french|spanish|german|italian|portuguese|chinese|japanese|korean|arabic|hindi|russian|dutch|swedish|norwegian|finnish|danish|polish|turkish|greek|hebrew|thai|vietnamese|indonesian|malay|swahili|latin|urdu|bengali|tamil|telugu|kannada|gujarati|marathi|punjabi)\b", re.IGNORECASE),
    re.compile(r"\b(?:from\s+\w+\s+to\s+\w+)\b", re.IGNORECASE),
]

CLASSIFICATION_KEYWORDS: Set[str] = {
    "classify", "categorize", "categorise", "label", "tag",
    "is this", "is it", "determine whether", "determine if",
    "positive or negative", "true or false", "yes or no",
    "which category", "which type", "which class",
    "sort into", "group into", "identify whether",
    "sentiment", "spam or not", "real or fake",
}

SUMMARIZATION_KEYWORDS: Set[str] = {
    "summarize", "summarise", "summary", "tldr", "tl;dr",
    "brief overview", "main points", "key takeaways",
    "in brief", "in short", "condense", "shorten",
    "recap", "digest", "abstract",
}

REWRITING_KEYWORDS: Set[str] = {
    "rewrite", "rephrase", "paraphrase", "reword",
    "rework", "revise", "edit", "improve", "correct",
    "fix the grammar", "make it", "turn this into",
    "simplify this", "formal version", "informal version",
    "more concise", "more detailed", "proofread",
}

EXPLANATION_KEYWORDS: Set[str] = {
    "explain", "describe", "elaborate", "clarify",
    "what does .+ mean", "why does", "why is", "why are",
    "how does", "how is", "how are", "how do",
    "what happens", "what causes", "what makes",
    "difference between", "differences between",
    "compare", "contrast", "distinguish",
    "pros and cons", "advantages and disadvantages",
    "benefits of", "drawbacks of",
}

REASONING_KEYWORDS: Set[str] = {
    "reason", "reasoning", "logic", "logical",
    "argue", "argument", "analyse", "analyze", "analysis",
    "evaluate", "assess", "judge", "critique", "critical",
    "infer", "inference", "deduce", "deduction",
    "conclude", "conclusion", "implication",
    "evidence", "support", "justify", "justification",
    "fallacy", "assumption", "hypothesis",
    "if then", "therefore", "hence", "thus",
    "given that", "assuming that", "it follows",
}

CHITCHAT_PATTERNS: List[re.Pattern] = [
    re.compile(r"^(?:hi|hello|hey|greetings|good morning|good afternoon|good evening|howdy|sup|yo)\b", re.IGNORECASE),
    re.compile(r"^(?:thanks|thank you|thx|ty|cheers)\b", re.IGNORECASE),
    re.compile(r"^(?:bye|goodbye|see you|later|ciao)\b", re.IGNORECASE),
    re.compile(r"^(?:how are you|how's it going|what's up|how do you do)\b", re.IGNORECASE),
    re.compile(r"^(?:nice|great|cool|awesome|ok|okay|sure|alright)\s*[!.]?\s*$", re.IGNORECASE),
    re.compile(r"^(?:yes|no|yeah|nah|yep|nope)\s*[!.]?\s*$", re.IGNORECASE),
]

UNSAFE_KEYWORDS: Set[str] = {
    "kill", "murder", "attack", "bomb", "weapon",
    "hack", "exploit", "steal", "fraud", "illegal",
    "drug", "narcotic", "suicide", "self-harm",
    "hate", "racist", "sexist", "slur",
    "porn", "nude", "nsfw", "explicit",
    "torture", "abuse", "violence", "threat",
}


# ═══════════════════════════════════════════════════════════════════════
# Complexity feature extractors
# ═══════════════════════════════════════════════════════════════════════

def _count_math_signals(text: str) -> int:
    """Count math-related signals in the text."""
    score = 0
    lowered = text.lower()

    # Keyword hits
    for kw in MATH_KEYWORDS:
        if kw in lowered:
            score += 1

    # Pattern hits
    for pat in MATH_PATTERNS:
        score += len(pat.findall(text))

    # Raw math operators (each one counts)
    score += len(re.findall(r"[=+\-*/^]", text))

    # Numbers — more numbers usually means more arithmetic
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    if len(numbers) >= 3:
        score += 2

    return score


def _count_code_signals(text: str) -> int:
    """Count code-related signals in the text."""
    score = 0
    lowered = text.lower()

    for kw in CODE_KEYWORDS:
        if kw in lowered:
            score += 1

    for pat in CODE_PATTERNS:
        score += len(pat.findall(text))

    return score


def _count_reasoning_signals(text: str) -> int:
    """Count reasoning-complexity signals."""
    score = 0
    lowered = text.lower()

    for kw in REASONING_KEYWORDS:
        if kw in lowered:
            score += 1

    # Multi-step markers
    for marker in ["first", "second", "third", "step 1", "step 2",
                    "then", "next", "finally", "additionally"]:
        if marker in lowered:
            score += 1

    # Conditional / logical connectives
    for conn in ["if", "because", "since", "although", "however",
                  "therefore", "hence", "thus", "given that"]:
        if f" {conn} " in f" {lowered} ":
            score += 1

    return score


def _estimate_non_english_ratio(text: str) -> float:
    """Estimate what fraction of the text is non-Latin / non-ASCII."""
    if not text:
        return 0.0
    non_ascii = sum(1 for ch in text if ord(ch) > 127 and not ch.isspace())
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    if alpha_chars == 0:
        return 0.0
    return non_ascii / alpha_chars


def _word_count(text: str) -> int:
    """Simple whitespace-based word count."""
    return len(text.split())


# ═══════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════

class IntentComplexityEstimator:
    """
    Heuristic-based intent classifier and complexity scorer.

    Usage::

        estimator = IntentComplexityEstimator()

        # Standalone — annotate a single segment dict
        annotated = estimator.estimate(segment_dict)

        # Batch — annotate all nodes from a Module-2 graph
        annotated_nodes = estimator.estimate_graph(graph)
    """

    # ── Band thresholds ────────────────────────────────────────────────
    SIMPLE_UPPER = 0.33
    MEDIUM_UPPER = 0.66

    def __init__(self) -> None:
        pass  # Stateless — no models to load for the heuristic version

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def estimate(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Annotate a single segment with intent, complexity, and reasons.

        Parameters
        ----------
        segment : dict
            Must contain at least ``"text"`` (str).
            May also contain ``"depends_on"`` (list), ``"depth"`` (int),
            ``"execution_mode"`` (str) from Module 2.

        Returns
        -------
        dict
            A **new** dict with all original keys plus:
              intent_label      : str
              intent_confidence : float
              complexity_score  : float   (0.0 – 1.0)
              complexity_band   : str     ("simple" / "medium" / "hard")
              unsafe_candidate  : bool
              reasons           : list[str]  — human-readable explanations
        """
        text = segment.get("text", "")
        reasons: List[str] = []

        # ---- 1. Feature extraction ------------------------------------
        features = self._extract_features(text, segment)

        # ---- 2. Intent classification ---------------------------------
        intent_label, intent_conf, intent_reasons = self._classify_intent(
            text, features
        )
        reasons.extend(intent_reasons)

        # ---- 3. Complexity estimation ---------------------------------
        complexity_score, complexity_reasons = self._score_complexity(
            text, features, intent_label, segment
        )
        reasons.extend(complexity_reasons)
        complexity_band = self._band(complexity_score)

        # ---- 4. Unsafe flag -------------------------------------------
        unsafe, unsafe_reasons = self._check_unsafe(text)
        reasons.extend(unsafe_reasons)

        # ---- Build result ---------------------------------------------
        result = dict(segment)   # shallow copy
        result.update({
            "intent_label": intent_label,
            "intent_confidence": round(intent_conf, 3),
            "complexity_score": round(complexity_score, 3),
            "complexity_band": complexity_band,
            "unsafe_candidate": unsafe,
            "reasons": reasons,
        })
        return result

    def estimate_graph(self, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Annotate every node in a Module-2 graph.

        Parameters
        ----------
        graph : dict
            The output of ``DependencyGraphBuilder.build()``.

        Returns
        -------
        list[dict]
            The ``graph["nodes"]`` list, each enriched with intent /
            complexity fields.
        """
        annotated: List[Dict[str, Any]] = []
        for node in graph["nodes"]:
            annotated.append(self.estimate(node))
        return annotated

    def estimate_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Annotate a flat segment list (e.g. direct Module-1 output)
        without a graph — useful for quick standalone testing.
        """
        return [self.estimate(seg) for seg in segments]

    # ──────────────────────────────────────────────────────────────────
    # Feature extraction
    # ──────────────────────────────────────────────────────────────────

    def _extract_features(
        self, text: str, segment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute all raw features used by intent + complexity."""
        lowered = text.lower()
        words = _word_count(text)

        return {
            "text_lower": lowered,
            "word_count": words,
            "char_count": len(text),
            "math_signals": _count_math_signals(text),
            "code_signals": _count_code_signals(text),
            "reasoning_signals": _count_reasoning_signals(text),
            "non_english_ratio": _estimate_non_english_ratio(text),
            "has_question_mark": text.strip().endswith("?"),
            "has_math_operator": bool(re.search(r"[=+\-*/^]", text)),
            "has_numbers": bool(re.search(r"\d", text)),
            "has_code_fence": "```" in text,
            "depth": segment.get("depth", 0),
            "depends_on": segment.get("depends_on", []),
            "is_dependent": bool(segment.get("depends_on", [])),
        }

    # ──────────────────────────────────────────────────────────────────
    # Intent classification
    # ──────────────────────────────────────────────────────────────────

    def _classify_intent(
        self,
        text: str,
        features: Dict[str, Any],
    ) -> Tuple[str, float, List[str]]:
        """
        Score each candidate intent and pick the best.

        Returns (intent_label, confidence, reasons).
        """
        lowered = features["text_lower"]
        scores: Dict[str, float] = {
            "retrieval": 0.0,
            "reasoning": 0.0,
            "math": 0.0,
            "code": 0.0,
            "generation": 0.0,
            "translation": 0.0,
            "classification": 0.0,
            "summarization": 0.0,
            "rewriting": 0.0,
            "explanation": 0.0,
            "chitchat": 0.0,
        }
        reasons: List[str] = []

        # ── Math ──────────────────────────────────────────────────────
        if features["math_signals"] > 0:
            scores["math"] += min(features["math_signals"] * 1.5, 10.0)
            reasons.append(
                f"math: {features['math_signals']} math signals detected"
            )
        if features["has_math_operator"] and features["has_numbers"]:
            scores["math"] += 2.0
            reasons.append("math: numbers + operators present")

        # ── Code ──────────────────────────────────────────────────────
        if features["code_signals"] > 0:
            scores["code"] += min(features["code_signals"] * 1.5, 10.0)
            reasons.append(
                f"code: {features['code_signals']} code signals detected"
            )
        if features["has_code_fence"]:
            scores["code"] += 5.0
            reasons.append("code: fenced code block present")

        # ── Retrieval ─────────────────────────────────────────────────
        for pat in RETRIEVAL_PATTERNS:
            if pat.search(text):
                scores["retrieval"] += 3.0
                reasons.append(f"retrieval: pattern match '{pat.pattern[:40]}'")
                break  # one match is enough

        for kw in RETRIEVAL_KEYWORDS:
            if kw in lowered:
                scores["retrieval"] += 2.0
                reasons.append(f"retrieval: keyword '{kw}'")
                break

        # ── Generation ────────────────────────────────────────────────
        for kw in GENERATION_KEYWORDS:
            if kw in lowered:
                scores["generation"] += 2.0
                reasons.append(f"generation: keyword '{kw}'")
                break

        # Boost if there's a creative-output word
        for creative in ["story", "poem", "essay", "haiku", "lyrics",
                         "post", "tweet", "letter", "email", "script"]:
            if creative in lowered:
                scores["generation"] += 1.5
                reasons.append(f"generation: creative output '{creative}'")
                break

        # ── Translation ───────────────────────────────────────────────
        for pat in TRANSLATION_PATTERNS:
            if pat.search(text):
                scores["translation"] += 4.0
                reasons.append(f"translation: pattern match")
                break

        if features["non_english_ratio"] > 0.3:
            scores["translation"] += 2.0
            reasons.append(
                f"translation: {features['non_english_ratio']:.0%} non-ASCII"
            )

        # ── Classification ────────────────────────────────────────────
        for kw in CLASSIFICATION_KEYWORDS:
            if kw in lowered:
                scores["classification"] += 3.0
                reasons.append(f"classification: keyword '{kw}'")
                break

        # ── Summarization ─────────────────────────────────────────────
        for kw in SUMMARIZATION_KEYWORDS:
            if kw in lowered:
                scores["summarization"] += 3.5
                reasons.append(f"summarization: keyword '{kw}'")
                break

        # ── Rewriting ─────────────────────────────────────────────────
        for kw in REWRITING_KEYWORDS:
            if kw in lowered:
                scores["rewriting"] += 3.0
                reasons.append(f"rewriting: keyword '{kw}'")
                break

        # ── Explanation ───────────────────────────────────────────────
        for kw in EXPLANATION_KEYWORDS:
            if kw in lowered:
                scores["explanation"] += 2.5
                reasons.append(f"explanation: keyword '{kw}'")
                break

        # ── Reasoning ─────────────────────────────────────────────────
        if features["reasoning_signals"] > 0:
            scores["reasoning"] += min(features["reasoning_signals"] * 1.2, 8.0)
            reasons.append(
                f"reasoning: {features['reasoning_signals']} reasoning signals"
            )

        # ── Chitchat ──────────────────────────────────────────────────
        for pat in CHITCHAT_PATTERNS:
            if pat.search(text):
                scores["chitchat"] += 5.0
                reasons.append("chitchat: greeting/farewell pattern match")
                break

        if features["word_count"] <= 5 and not features["has_question_mark"]:
            scores["chitchat"] += 1.0

        # ── Pick winner ──────────────────────────────────────────────
        best_intent = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_intent]

        # If nothing scored above zero, default to "generation"
        if best_score == 0.0:
            best_intent = "generation"
            best_score = 1.0
            reasons.append("intent: no strong signal — defaulting to generation")

        # Confidence = best_score / (best_score + second_best) — a
        # simple normalised margin.
        sorted_scores = sorted(scores.values(), reverse=True)
        second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        denom = best_score + second_best
        confidence = best_score / denom if denom > 0 else 1.0

        reasons.insert(0, f"intent → {best_intent} (confidence {confidence:.2f})")

        return best_intent, confidence, reasons

    # ──────────────────────────────────────────────────────────────────
    # Complexity scoring
    # ──────────────────────────────────────────────────────────────────

    def _score_complexity(
        self,
        text: str,
        features: Dict[str, Any],
        intent: str,
        segment: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """
        Produce a complexity score ∈ [0, 1] from multiple weighted signals.

        Returns (score, reasons).
        """
        raw = 0.0
        reasons: List[str] = []

        # ── 1. Intent-based baseline ──────────────────────────────────
        INTENT_BASELINES: Dict[str, float] = {
            "math": 0.55,
            "code": 0.55,
            "reasoning": 0.50,
            "explanation": 0.35,
            "generation": 0.30,
            "translation": 0.30,
            "classification": 0.25,
            "summarization": 0.25,
            "rewriting": 0.25,
            "retrieval": 0.15,
            "chitchat": 0.05,
        }
        baseline = INTENT_BASELINES.get(intent, 0.30)
        raw += baseline
        reasons.append(f"complexity baseline for '{intent}': {baseline:.2f}")

        # ── 2. Math density bonus ─────────────────────────────────────
        math_sig = features["math_signals"]
        if math_sig >= 5:
            raw += 0.25
            reasons.append(f"math density high ({math_sig} signals): +0.25")
        elif math_sig >= 2:
            raw += 0.12
            reasons.append(f"math density medium ({math_sig} signals): +0.12")

        # ── 3. Code density bonus ─────────────────────────────────────
        code_sig = features["code_signals"]
        if code_sig >= 5:
            raw += 0.25
            reasons.append(f"code density high ({code_sig} signals): +0.25")
        elif code_sig >= 2:
            raw += 0.10
            reasons.append(f"code density medium ({code_sig} signals): +0.10")

        # ── 4. Reasoning density bonus ────────────────────────────────
        reas_sig = features["reasoning_signals"]
        if reas_sig >= 4:
            raw += 0.15
            reasons.append(f"reasoning density high ({reas_sig} signals): +0.15")
        elif reas_sig >= 2:
            raw += 0.08
            reasons.append(f"reasoning density medium ({reas_sig} signals): +0.08")

        # ── 5. Length bonus (longer ≈ harder, up to a point) ──────────
        wc = features["word_count"]
        if wc >= 80:
            raw += 0.10
            reasons.append(f"long segment ({wc} words): +0.10")
        elif wc >= 40:
            raw += 0.05
            reasons.append(f"medium segment ({wc} words): +0.05")
        elif wc <= 5:
            raw -= 0.05
            reasons.append(f"very short segment ({wc} words): -0.05")

        # ── 6. Dependency depth bonus ─────────────────────────────────
        depth = features.get("depth", 0)
        if depth >= 2:
            raw += 0.15
            reasons.append(f"dependency depth {depth}: +0.15")
        elif depth == 1:
            raw += 0.05
            reasons.append(f"dependency depth {depth}: +0.05")

        # ── 7. Dependent segment bonus ────────────────────────────────
        if features["is_dependent"]:
            raw += 0.08
            reasons.append("segment is dependent on prior: +0.08")

        # ── 8. Question complexity ────────────────────────────────────
        if features["has_question_mark"]:
            lowered = features["text_lower"]
            # Simple factual questions → reduce
            if any(lowered.startswith(w) for w in
                   ["who is", "what is", "when was", "where is"]):
                raw -= 0.05
                reasons.append("simple factual question: -0.05")
            # "How many" / "how much" with numbers → likely math
            elif re.search(r"\bhow\s+(?:many|much)\b", lowered):
                raw += 0.08
                reasons.append("quantitative question: +0.08")

        # ── 9. Multi-step markers ─────────────────────────────────────
        lowered = features["text_lower"]
        step_count = sum(1 for m in
                         ["first", "second", "third", "step 1", "step 2",
                          "step 3", "then", "finally"]
                         if m in lowered)
        if step_count >= 3:
            raw += 0.15
            reasons.append(f"multi-step markers ({step_count}): +0.15")
        elif step_count >= 2:
            raw += 0.08
            reasons.append(f"multi-step markers ({step_count}): +0.08")

        # ── Clamp to [0, 1] ──────────────────────────────────────────
        final = max(0.0, min(1.0, raw))
        reasons.append(f"complexity → {final:.3f} ({self._band(final)})")

        return final, reasons

    # ──────────────────────────────────────────────────────────────────
    # Safety pre-check (lightweight keyword flag)
    # ──────────────────────────────────────────────────────────────────

    def _check_unsafe(self, text: str) -> Tuple[bool, List[str]]:
        """
        Quick keyword scan for unsafe content.
        This is NOT the full safety gate (Module 6) — just a flag for
        the router to know that a deeper check is warranted.
        """
        lowered = text.lower()
        reasons: List[str] = []
        found: List[str] = []

        for kw in UNSAFE_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", lowered):
                found.append(kw)

        if found:
            reasons.append(
                f"unsafe_candidate: keywords {found[:5]} detected"
            )
            return True, reasons

        return False, reasons

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _band(self, score: float) -> str:
        """Map a [0, 1] score to a band label."""
        if score <= self.SIMPLE_UPPER:
            return "simple"
        elif score <= self.MEDIUM_UPPER:
            return "medium"
        else:
            return "hard"

    # ──────────────────────────────────────────────────────────────────
    # Pretty-print helper
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def print_annotations(
        annotated_nodes: List[Dict[str, Any]],
        show_reasons: bool = True,
    ) -> None:
        """Print a human-readable summary of all annotated segments."""
        print("=" * 70)
        print("INTENT & COMPLEXITY ANNOTATIONS")
        print("=" * 70)

        for node in annotated_nodes:
            sid = node.get("segment_id", node.get("id", "?"))
            intent = node.get("intent_label", "?")
            conf = node.get("intent_confidence", 0)
            score = node.get("complexity_score", 0)
            band = node.get("complexity_band", "?")
            unsafe = node.get("unsafe_candidate", False)
            text = node.get("text", "")

            print(f"\n  [{sid}] \"{text[:75]}{'…' if len(text) > 75 else ''}\"")
            print(f"       intent        = {intent} (confidence {conf:.2f})")
            print(f"       complexity    = {score:.3f} ({band})")
            if unsafe:
                print(f"       ⚠ unsafe_candidate = True")
            if show_reasons and "reasons" in node:
                print(f"       reasons:")
                for r in node["reasons"]:
                    print(f"         • {r}")

        print("\n" + "=" * 70)
