import pickle
import socket
import logging

from flask import Flask, request, jsonify

from model_training import run_model_training
from data_ingestion import clean_text, run_setup_database


# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    filename="sentiment.log",
    filemode="a",
    format="%(asctime)s - %(lineno)d - %(name)s - %(message)s"
)


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    # Load model and vectorizer
    model = pickle.load(open("sentiment_model.pkl", "rb"))
    tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    
    # get json from request
    data = request.get_json()
    if "review_text" not in data:
        return jsonify({"error": "Missing 'review_text' field"}), 400
    
    review_text = data["review_text"]
    cleaned_review = clean_text(review_text)
    text_vectorized = tfidf_vectorizer.transform([cleaned_review])
    
    prediction = model.predict(text_vectorized)
    sentiment = "Positive" if prediction[0] == 0 else "Negative"
    
    return jsonify({"sentiment_prediction": sentiment})


if __name__ == "__main__":
    run_setup_database()
    run_model_training()
    
    host = socket.gethostbyname(socket.gethostname())
    port = 5000
    print(f"Flask app running at: http://{host}:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)