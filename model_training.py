import pickle
import logging
import sqlite3
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    filename="sentiment.log",
    filemode="a",
    format="%(asctime)s - %(lineno)d - %(name)s - %(message)s"
)


# Model training function
def model_training(db_file, model_file, vectorizer_file):
    logging.info("Starting Model Training")
    
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query("SELECT * FROM imdb_reviews", conn)
    conn.close()
    
    # Independent and dependent feature
    X = df["review_text"]
    y = df["sentiment"].apply(lambda x: 0 if x == "positive" else 1)    # positive = 0, negative = 1
    
    # Train - Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tf = tfidf_vectorizer.transform(X_test)
    
    # Define Models
    models = {
        # "Support Vector Classifier": SVC(),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(),
    }
    
    # Store model performance
    model_performance = {}
    
    for name, model in models.items():
        logging.info(f"Model : {name}")
        
        # Training
        model.fit(X_train_tf, y_train)
        
        # Prediction
        y_pred_train = model.predict(X_train_tf)
        y_pred_test = model.predict(X_test_tf)
        
        # Accuracy
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        logging.info(f"Training Accuracy : {train_accuracy}")
        logging.info(f"Testing Accuracy : {test_accuracy}")
        
        # Store test accuracy
        model_performance[name] = test_accuracy
        logging.info("="*70)
    
    # Find the best model
    best_model_name = max(model_performance, key=model_performance.get)
    best_model = models[best_model_name]
    logging.info(f"Best Model = {best_model_name}\t Accuracy = {model_performance[best_model_name]: .4f}")
    
    # Save model and vectorizer
    pickle.dump(best_model, open(model_file, "wb"))
    pickle.dump(tfidf_vectorizer, open(vectorizer_file, "wb"))
    logging.info("Model and vectorizer saved successfully.")


# Run model_training function
def run_model_training():
    model_training("imdb_reviews.db", "sentiment_model.pkl", "tfidf_vectorizer.pkl")