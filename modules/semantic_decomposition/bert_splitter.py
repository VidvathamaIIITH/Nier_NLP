from transformers import pipeline

class BertSplitter:

    def __init__(self):

        self.model = pipeline(
            "token-classification",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )

    def predict_boundaries(self, text):

        result = self.model(text)

        boundaries = []

        for r in result:

            if r["score"] > 0.8:
                boundaries.append(r["start"])

        return boundaries