from neuro_symbolic import TextNeuralProcessor, TextSymbolicProcessor

# Initialize the neural and symbolic processors for text
text_neural_processor = TextNeuralProcessor(model_path="sentiment_model.pth")
text_symbolic_processor = TextSymbolicProcessor(knowledge_base="policy_kb.json")

# Customer input
customer_query = "Can I return a product after 30 days?"

# Analyze sentiment and extract key information
sentiment, keywords = text_neural_processor.process_text(customer_query)

# Use symbolic reasoning to generate a response based on company policies
response = text_symbolic_processor.generate_response(keywords, sentiment)

print(f"Customer Service Response: {response}")