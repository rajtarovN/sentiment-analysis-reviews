import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

def compute_detailed_metrics(y_test, results):
    rows = []
    for model_name, data in results.items():
        y_pred = data["y_pred"]
        rows.append({
            "Model": model_name,
            "Accuracy": data["accuracy"],
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred)
        })

    metrics_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    metrics_df.to_csv("../results/detailed_metrics.csv", index=False)
    print(metrics_df.to_string(index=False))
    return metrics_df


def plot_detailed_metrics(metrics_df):
    metrics_to_plot = ["Precision", "Recall", "F1-score"]
    x = np.arange(len(metrics_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i * width, metrics_df[metric], width, label=metric)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_df["Model"])
    ax.set_ylabel("Vrednost")
    ax.set_ylim(0, 1)
    ax.set_title("Comparing model: Precision, Recall i F1-mera")
    ax.legend()

    plt.tight_layout()
    plt.savefig("../results/detailed_metrics_chart.png", dpi=150)
    plt.close()
def load_data():
    with open("../results/eval_data.pkl", "rb") as f:
        eval_data = pickle.load(f)

    with open("../results/trained_models.pkl", "rb") as f:
        trained_models = pickle.load(f)

    with open("../results/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return eval_data, trained_models, vectorizer

def best_model(vectorizer, best_model_name, trained_models):
    feature_names = np.array(vectorizer.get_feature_names_out())

    if best_model_name in ["Logistic Regression", "SVM (Linear)"]:
        model = trained_models[best_model_name]
        coefs = model.coef_[0]

        top_positive_idx = np.argsort(coefs)[-15:][::-1]
        top_negative_idx = np.argsort(coefs)[:15]

        top_positive_words = feature_names[top_positive_idx]
        top_negative_words = feature_names[top_negative_idx]

        print("\nTop 15 words connected with positive recensions:")
        print(", ".join(top_positive_words))

        print("\nTop 15 words connected with negative recensions:")
        print(", ".join(top_negative_words))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].barh(top_positive_words[::-1], coefs[top_positive_idx][::-1], color="#55A868")
        axes[0].set_title("Words connected with positive sentiment")

        axes[1].barh(top_negative_words[::-1], coefs[top_negative_idx][::-1], color="#C44E52")
        axes[1].set_title("Words connected with negative sentiment")

        plt.tight_layout()

def main():
    eval_data, trained_models, vectorizer = load_data()
    y_test  =eval_data["y_test"]
    results= eval_data["results"]

    model_names = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in model_names]
    metrics_df = compute_detailed_metrics(y_test, results)
    plot_detailed_metrics(metrics_df)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, accuracies, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.ylabel("Accuracy")
    plt.title("Comparison of models according to classification accuracy")
    plt.ylim(0, 1)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc:.3f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("../results/comparison_chart.png", dpi=150)
    plt.close()

    best_model_name = max(results, key=lambda m: results[m]["accuracy"])
    print(f"\nBest model: {best_model_name}")

    y_pred_best = results[best_model_name]["y_pred"]

    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Confusion matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig("../results/confusion_matrix.png", dpi=150)
    plt.close()

    best_model(vectorizer, best_model_name, trained_models)

if __name__=="__main__":
    main()