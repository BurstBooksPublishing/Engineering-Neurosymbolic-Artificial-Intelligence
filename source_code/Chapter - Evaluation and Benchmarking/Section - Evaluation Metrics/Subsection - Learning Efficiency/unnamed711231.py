import nltk
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Sample data: list of texts and their labels
texts = ["I love this product", "I hate this product", "This is the best", "This is the worst"]
labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Tokenizing text
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=10)

# Building a simple LSTM model
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(data, labels, epochs=10)

# Symbolic rules for refining output
def refine_sentiment(prediction, text):
    if 'love' in text or 'best' in text:
        return 1  # Positive sentiment
    elif 'hate' in text or 'worst' in text:
        return 0  # Negative sentiment
    return round(prediction)

# Testing the neuro-symbolic system
test_text = "I think this is the best product"
test_seq = tokenizer.texts_to_sequences([test_text])
test_data = pad_sequences(test_seq, maxlen=10)

prediction = model.predict(test_data)[0][0]
refined_prediction = refine_sentiment(prediction, test_text)

print("Refined Prediction:", refined_prediction)