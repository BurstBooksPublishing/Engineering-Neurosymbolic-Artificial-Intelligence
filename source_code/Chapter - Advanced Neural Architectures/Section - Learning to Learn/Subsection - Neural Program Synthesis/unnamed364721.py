from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Encode input context
inputs = tokenizer.encode("Write a function in Python that adds two numbers", return_tensors='pt')

# Generate code autoregressively
outputs = model.generate(inputs, max_length=50, num_return_sequences=5)

print("Generated Code Snippets:")
for i, output in enumerate(outputs):
    print(f"{i+1}: {tokenizer.decode(output)}")