from flask import Flask
import re
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import nltk
import tensorflow as tf

from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import GPT2LMHeadModel, GPT2Tokenizer

nltk.download("punkt")
nltk.download("stopwords")

tokenizer = Tokenizer(num_words = 10000)
t_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

df = pd.read_csv("./mental_health.csv")

model = tf.keras.models.load_model("./Sentiment.h5")
t_model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=t_tokenizer.eos_token_id)

sentiment = ctrl.Antecedent(np.arange(-1, 1.01, 0.01), 'sentiment')

labels = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
sentiment.automf(names=labels)

labels_consequent = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
consequent = ctrl.Consequent(np.arange(0, 101, 1), 'label')
consequent.automf(names=labels_consequent)

rules = [
    ctrl.Rule(sentiment['very_negative'], consequent['very_negative']),
    ctrl.Rule(sentiment['negative'], consequent['negative']),
    ctrl.Rule(sentiment['neutral'], consequent['neutral']),
    ctrl.Rule(sentiment['positive'], consequent['positive']),
    ctrl.Rule(sentiment['very_positive'], consequent['very_positive'])
]

sentiment_ctrl = ctrl.ControlSystem(rules)
sentiment_prediction = ctrl.ControlSystemSimulation(sentiment_ctrl)

app = Flask(__name__)

column_names = ["text"]

def preprocess_text(text):
    text = text.lower # Change to lowercase
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuaions
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words] # Remove Stopwords
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

df['text'] = preprocess_text(df['text'])

tokenizer.fit_on_text(df['text'])

def fuzzy_layer(sentiment):
    lstm_sentiment_predictions = sentiment

    defuzzified_labels = np.empty_like(lstm_sentiment_predictions)
    sentiment_ctrl = ctrl.ControlSystem(rules)
    for i, lstm_sentiment_prediction in enumerate(lstm_sentiment_predictions):
        sentiment_prediction = ctrl.ControlSystemSimulation(sentiment_ctrl)
        sentiment_prediction.input['sentiment'] = lstm_sentiment_prediction
        sentiment_prediction.compute()
        defuzzified_labels[i] = sentiment_prediction.output['label']

    labels = []

    for label in defuzzified_labels:
        if label <= 12.4:
            labels.append("Advice for someone who defintely doesn't have a mental illness: ") # Very-negative
        elif 12.5 <= label <= 37.4:
            labels.append("Advice for someone who doesn't not have a mental illness: ") # Negative
        elif 37.5 <= label <= 62.4:
            labels.append("Neutral") # Neutral
        elif 62.5 <= label <= 87.4:
            labels.append("Advice for someone who has a mental illness: ") # Positive
        else:
            labels.append("Advice for someone who defintely has a mental illness: ") # Very-positive

    return labels

def generate_response(prompt):
    input_ids = t_tokenizer.encode(prompt, return_tensors='pt')
    output = t_model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.route('/')
def home():
    pass

@app.route('/speak')
def speak(input):
    text = []
    text.append(input)
    text_array = [text]

    data = pd.DataFrame(text_array, columns=column_names)
    data = data["text"].apply(preprocess_text)
    sequences = tokenizer.texts_to_sequence(data)
    data = pad_sequences(sequences, maxlen=200)

    predictions = model.predict(data)

    sentiment_scores = predictions[:, 1] - predictions[:, 0]

    label = fuzzy_layer(sentiment_scores)

    return generate_response(label[-1])