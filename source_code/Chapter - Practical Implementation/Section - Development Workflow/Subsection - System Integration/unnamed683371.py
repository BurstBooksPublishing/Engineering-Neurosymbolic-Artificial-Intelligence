from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Encode text
input_text = "I had a long day at work."
encoded_input = tokenizer(input_text, return_tensors='pt')

# Predict sentiment
with torch.no_grad():
    outputs = model(encoded_input)

# Symbolic reasoning to generate a response
if torch.argmax(outputs.logits) == 1:  # Assuming class 1 is positive sentiment
    response = "Glad you had a good day! Want to relax with some music?"
else:
    response = "Sorry to hear that. Would you like to talk about it?"

print(response)