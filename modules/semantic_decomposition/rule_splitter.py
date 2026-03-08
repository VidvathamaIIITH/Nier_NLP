import spacy

class RuleSplitter:

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")

    def _looks_like_clause(self, fragment):

        doc = self.nlp(fragment)
        return any(token.pos_ in ["VERB", "AUX"] for token in doc)

    def split(self, text):

        doc = self.nlp(text)

        segments = []

        # Step 1 — sentence segmentation
        for sent in doc.sents:

            sentence = sent.text.strip()

            # Step 2 — split multi-task conjunctions
            if " and " in sentence.lower():

                parts = sentence.split(" and ")

                if len(parts) == 2 and self._looks_like_clause(parts[0]) and self._looks_like_clause(parts[1]):
                    for p in parts:
                        segments.append(p.strip())
                else:
                    segments.append(sentence)

            else:
                segments.append(sentence)

        return segments