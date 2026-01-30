from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def chatbot_answer(question, text):
    sentences = text.split(".")
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences + [question])
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    idx = similarity.argmax()
    return sentences[idx]
