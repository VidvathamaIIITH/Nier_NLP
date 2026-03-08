from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download("punkt")

class EmbeddingSplitter:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def split(self, text):

        sentences = nltk.sent_tokenize(text)

        if len(sentences) <= 1:
            return sentences

        embeddings = self.model.encode(sentences)

        segments = [sentences[0]]

        for i in range(1,len(sentences)):

            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[i-1]]
            )[0][0]

            if sim < 0.55:
                segments.append(sentences[i])
            else:
                segments[-1] += " " + sentences[i]

        return segments