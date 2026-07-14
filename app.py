from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = tf.keras.models.load_model("imdb_lstm_model.h5")

word_index = imdb.get_word_index()

maxlen = 200


def preprocess(text):

    text = text.lower().split()

    sequence = []

    for word in text:
        if word in word_index:
            sequence.append(word_index[word] + 3)
        else:
            sequence.append(2)

    padded = pad_sequences([sequence],
                           maxlen=maxlen)

    return padded


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    review = request.form["review"]

    processed = preprocess(review)

    prediction = model.predict(processed)[0][0]

    if prediction >= 0.5:
        result = "😊 Positive Review"
    else:
        result = "😞 Negative Review"

    return render_template("index.html",
                           prediction=result)


if __name__ == "__main__":
    app.run(debug=True)