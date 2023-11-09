from flask import Flask, request, jsonify
import re
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Download necessary NLTK models
nltk.download("punkt")
nltk.download("stopwords")

# Load the tokenizer and model
tokenizer = Tokenizer(num_words=10000)
t_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = load_model("./Sentiment.h5")
t_model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=t_tokenizer.eos_token_id)

app = Flask(__name__)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Read the dataset
df = pd.read_csv("./mental_health.csv")
df['text'] = df['text'].apply(preprocess_text)

# Fit the tokenizer on the texts
tokenizer.fit_on_texts(df['text'].values)

@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    input_text = data['input']

    # Preprocess the input text
    processed_text = preprocess_text(input_text)

    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=200)

    # Predict sentiment
    prediction = model.predict(padded_sequence)
    sentiment_score = prediction[:, 1] - prediction[:, 0]

    # Generate a response using GPT-2
    input_ids = t_tokenizer.encode(processed_text, return_tensors='pt')
    output = t_model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    response = t_tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
