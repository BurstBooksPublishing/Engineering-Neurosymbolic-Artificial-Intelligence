import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample data: descriptions and their corresponding diseases
data = [
    ("fever, cough, and shortness of breath", "COVID-19"),
    ("joint pain, rash, and headache", "Lyme Disease"),
    ("abdominal pain, vomiting, and fever", "Gastroenteritis")
]

# Extracting features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([item[0] for item in data])

# Symbolic rules as a simple dictionary
rules = {
    "COVID-19": ["fever", "cough", "breath"],
    "Lyme Disease": ["joint", "rash", "headache"],
    "Gastroenteritis": ["abdominal", "vomiting", "fever"]
}

def diagnose(symptoms):
    # Convert symptoms to vector
    symptoms_vec = vectorizer.transform([symptoms])
    
    # Find the most similar disease description
    similarities = cosine_similarity(symptoms_vec, X)
    most_similar_idx = np.argmax(similarities)
    predicted_disease = data[most_similar_idx][1]
    
    # Reasoning with symbolic rules
    relevant_symptoms = rules[predicted_disease]
    if all(word in symptoms for word in relevant_symptoms):
        return predicted_disease
    else:
        return "Uncertain"

# Example usage
symptoms_input = "I have a fever and a cough"
diagnosed_disease = diagnose(symptoms_input)

print(f"Diagnosed Disease: {diagnosed_disease}")