import numpy as np

class MemoryNetwork:
    def __init__(self, memory_size, embedding_dim):
        self.memory = np.random.randn(memory_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def remember(self, facts):
        # Simple example: store facts by embedding them into memory
        for i, fact in enumerate(facts):
            self.memory[i % len(self.memory)] = self.embed(fact)

    def embed(self, fact):
        # Convert fact to vector (simple hash-based embedding for demonstration)
        return np.array([hash(word) % self.embedding_dim for word in fact.split()])

    def answer(self, query):
        # Answer query based on memory contents
        query_vec = self.embed(query)
        scores = np.dot(self.memory, query_vec)
        return np.argmax(scores)

# Example usage
memory_network = MemoryNetwork(memory_size=10, embedding_dim=5)
memory_network.remember(["the sky is blue", "grass is green"])
query = "what color is the sky?"
answer_index = memory_network.answer(query)
print(f"Answer index: {answer_index}")