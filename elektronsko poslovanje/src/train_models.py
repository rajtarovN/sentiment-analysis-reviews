import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    df = pd.read_csv("../data/reviews_clean.csv")
    df = df.dropna(subset=["clean_text"])
    X = df["clean_text"]
    y = df["label"]

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

def train_model( X_train, X_test, y_train, y_test ):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM (Linear)": LinearSVC()
    }
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])

        print(f"Accuracy: {acc:.4f}")
        print(report)

        results[name] = {
            "accuracy": acc,
            "y_pred": y_pred
        }
        trained_models[name] = model
    return results, vectorizer, trained_models


def save_data(results, vectorizer, trained_models, y_test):
    with open("../results/results.pkl", "wb") as f:
        pickle.dump(results, f)
    os.makedirs("results", exist_ok=True)
    with open("../results/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("../results/trained_models.pkl", "wb") as f:
        pickle.dump(trained_models, f)
    with open("../results/eval_data.pkl", "wb") as f:
        pickle.dump({
            "y_test": y_test,
            "results": results
        }, f)

def compare_models(results):
    comparison_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": [results[m]["accuracy"] for m in results]
    }).sort_values("Accuracy", ascending=False)

    comparison_df.to_csv("../results/metrics.csv", index=False)
    print("Comparison models")
    print(comparison_df)

def main():
    X_train, X_test, y_train, y_test=load_data()
    results, vectoriser, train_models= train_model(  X_train, X_test, y_train, y_test)
    save_data(results, vectoriser, train_models, y_test)
    compare_models(results)

if __name__ == "__main__":
    main()