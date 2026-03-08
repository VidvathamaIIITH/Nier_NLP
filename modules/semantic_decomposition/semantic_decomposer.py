import re
from typing import Dict, List

import spacy


class SemanticDecomposer:

    SPLIT_CONJUNCTIONS = {"and", "then", "also", "but"}
    DEPENDENCY_MARKERS = {
        "it",
        "they",
        "them",
        "that",
        "those",
        "these",
        "this",
        "former",
        "latter",
        "result",
        "answer",
    }
    ACTION_PREFIXES = {
        "find",
        "name",
        "write",
        "choose",
        "list",
        "identify",
        "explain",
        "compare",
        "tell",
        "describe",
        "solve",
        "output",
        "summarize",
        "translate",
        "generate",
        "give",
        "compute",
        "calculate",
        "use",
        "return",
    }

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = self._load_nlp(model_name)

    def _load_nlp(self, model_name: str):
        try:
            return spacy.load(model_name)
        except OSError:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp

    def _normalize_fragment(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip(" ,;:-")

    def _looks_like_task(self, fragment: str) -> bool:
        normalized = self._normalize_fragment(fragment)
        if len(normalized) < 3:
            return False

        doc = self.nlp(normalized)
        if not len(doc):
            return False

        first_token = doc[0].text.lower()
        token_count = len([token for token in doc if not token.is_space and not token.is_punct])
        root = next((token for token in doc if token.head == token), None)
        has_subject = any(token.dep_ in {"nsubj", "nsubjpass", "expl"} for token in doc)

        has_predicate = any(token.pos_ in {"VERB", "AUX"} for token in doc)
        has_math_signal = bool(re.search(r"[=+\-*/]", normalized))
        has_question_signal = normalized.endswith("?") or any(
            token.lower_ in {"who", "what", "when", "where", "why", "how"}
            for token in doc[:2]
        )
        has_imperative_shape = (
            root is not None
            and root.pos_ in {"VERB", "AUX"}
            and (root.i == 0 or first_token in self.ACTION_PREFIXES)
            and token_count >= 2
        )
        has_statement_shape = root is not None and root.pos_ in {"VERB", "AUX"} and has_subject

        if first_token in self.ACTION_PREFIXES and token_count < 2 and not has_math_signal:
            return False

        return has_math_signal or has_question_signal or has_imperative_shape or has_statement_shape or (
            has_predicate and first_token in self.ACTION_PREFIXES and token_count >= 2
        )

    def _dependency_phrase_match(self, fragment: str) -> bool:
        lowered = fragment.lower()
        phrases = [
            "use that",
            "use it",
            "that result",
            "the result",
            "the answer",
            "based on that",
            "using the above",
            "using that",
            "then use",
        ]
        return any(phrase in lowered for phrase in phrases)

    def _is_dependent(self, fragment: str) -> bool:
        normalized = self._normalize_fragment(fragment)
        if not normalized:
            return False

        if self._dependency_phrase_match(normalized):
            return True

        doc = self.nlp(normalized)
        if not len(doc):
            return False

        first = doc[0].text.lower()
        if first in self.DEPENDENCY_MARKERS:
            return True

        return any(token.dep_ in {"nsubj", "dobj", "pobj"} and token.lower_ in self.DEPENDENCY_MARKERS for token in doc)

    def _candidate_split_indices(self, sentence: str, doc) -> List[tuple[int, str]]:
        indices = []

        for token in doc:
            if token.dep_ == "cc" and token.text.lower() in self.SPLIT_CONJUNCTIONS:
                indices.append((token.idx, token.text.lower()))

        for match in re.finditer(r"\s*;\s*", sentence):
            indices.append((match.start(), ";"))

        action_prefix_pattern = "|".join(sorted(self.ACTION_PREFIXES))
        for match in re.finditer(
            rf",\s*(then\s+)?(?=(?:{action_prefix_pattern})\b)",
            sentence,
            flags=re.IGNORECASE,
        ):
            indices.append((match.start(), ","))

        return sorted(set(indices), key=lambda item: item[0])

    def split_conjunction(self, sentence: str) -> List[str]:
        sentence = self._normalize_fragment(sentence)
        if not sentence:
            return []

        doc = self.nlp(sentence)

        for split_idx, splitter in self._candidate_split_indices(sentence, doc):
            left = self._normalize_fragment(sentence[:split_idx])
            right = self._normalize_fragment(sentence[split_idx:])

            if right:
                right = re.sub(r"^(and|then|also|but)\b", "", right, flags=re.IGNORECASE).strip(" ,;:-")

            if splitter in {"then", "also"} and not re.match(r"^(find|name|write|choose|list|identify|explain|compare|tell|describe|solve|output|summarize|translate|generate|give|compute|calculate|use|return)\b", right, flags=re.IGNORECASE):
                continue

            if self._looks_like_task(left) and self._looks_like_task(right):
                left_parts = self.split_conjunction(left)
                right_parts = self.split_conjunction(right)
                return left_parts + right_parts

        return [sentence]

    def decompose(self, text: str) -> List[Dict[str, object]]:
        if "```" in text:
            cleaned_text = self._normalize_fragment(text)
            return [
                {
                    "id": 0,
                    "text": cleaned_text,
                    "depends_on_previous": False,
                    "execution": "parallel",
                }
            ]

        doc = self.nlp(text)
        raw_segments = []

        for sent in doc.sents:
            sentence = self._normalize_fragment(sent.text)
            if len(sentence) < 3:
                continue

            raw_segments.extend(self.split_conjunction(sentence))

        tasks = []

        for index, segment in enumerate(raw_segments):
            cleaned_segment = self._normalize_fragment(segment)
            if not cleaned_segment:
                continue

            dependent = index > 0 and self._is_dependent(cleaned_segment)
            tasks.append(
                {
                    "id": len(tasks),
                    "text": cleaned_segment,
                    "depends_on_previous": dependent,
                    "execution": "sequential" if dependent else "parallel",
                }
            )

        return tasks