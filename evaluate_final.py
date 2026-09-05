

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


FINE_TUNED_MODEL_PATH = "./final"   # <-- change this

# =======================================================================
# 1. LOAD DATASET
# =======================================================================
df = pd.read_csv("test.tsv", sep="\t")
df["sentiment"] = df["sentiment"].str.capitalize()   # normalize to Positive/Negative/Neutral
print(f"Loaded {len(df)} labeled code-mixed sentences")
print(df["sentiment"].value_counts())

LABELS = ["Negative", "Neutral", "Positive"]

# =======================================================================
# 2. ESTIMATE CODE-MIXING LEVEL PER SENTENCE (dataset has no mixing_level column)
#    Heuristic: fraction of words that are recognized English dictionary words.
#    low = mostly Nepali, high = heavily English-mixed
# =======================================================================
import nltk
try:
    nltk.data.find("corpora/words")
except LookupError:
    nltk.download("words")
from nltk.corpus import words as nltk_words
ENGLISH_VOCAB = set(w.lower() for w in nltk_words.words())


def estimate_mixing_level(text):
    tokens = [t.strip(".,!?😍🤣😂🥱❤😭🤩😘").lower() for t in str(text).split()]
    tokens = [t for t in tokens if t.isalpha()]
    if not tokens:
        return "low"
    eng_ratio = sum(1 for t in tokens if t in ENGLISH_VOCAB) / len(tokens)
    if eng_ratio < 0.2:
        return "low"
    elif eng_ratio < 0.5:
        return "medium"
    else:
        return "high"


df["mixing_level"] = df["text"].apply(estimate_mixing_level)
print("\nEstimated mixing level distribution:")
print(df["mixing_level"].value_counts())

# =======================================================================
# 3. LOAD MODELS
# =======================================================================
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

print("\nLoading models...")

pipelines = {}

# Model A: YOUR fine-tuned XLM-R (the specialized model, like Sagarmatha V4 in the ASR project)
tok = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
mdl = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH)
pipelines["Fine-tuned XLM-R (ours)"] = pipeline(
    "sentiment-analysis", model=mdl, tokenizer=tok
)
print("id2label for our fine-tuned model:", mdl.config.id2label)

# Model B: general multilingual sentiment model (XLM-R, twitter-trained, many languages)
pipelines["XLM-R Twitter Sentiment (general)"] = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

# Model C: general multilingual BERT sentiment (1-5 star rating model)
pipelines["mBERT Multilingual (general)"] = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Model D: Nepal-aware baseline - mBERT fine-tuned on Nepali sentiment
# (documented 3-class label order: 0=Negative, 1=Positive, 2=Neutral)

# Neutral task, so it's been replaced with this confirmed 3-class model.
nepali_bert_tok = AutoTokenizer.from_pretrained("dpkrm/NepaliSentimentAnalysis")
nepali_bert_mdl = AutoModelForSequenceClassification.from_pretrained("dpkrm/NepaliSentimentAnalysis")
pipelines["mBERT Nepali Sentiment (Nepal-aware)"] = pipeline(
    "sentiment-analysis", model=nepali_bert_mdl, tokenizer=nepali_bert_tok
)
print("id2label for Nepal-aware mBERT baseline:", nepali_bert_mdl.config.id2label)

# =======================================================================
# 4. NORMALIZE MODEL OUTPUT LABELS -> Negative/Neutral/Positive
# =======================================================================
def normalize_label(model_name, raw_label):
    raw = raw_label.lower()

    if model_name == "Fine-tuned XLM-R (ours)":
        if raw in ("positive", "negative", "neutral"):
            return raw.capitalize()
        manual_map = {
            "label_0": "Negative",
            "label_1": "Neutral",
            "label_2": "Positive",
        }
        return manual_map.get(raw, raw_label)

    if model_name == "XLM-R Twitter Sentiment (general)":
        return raw.capitalize()  # returns positive/negative/neutral

    if model_name == "mBERT Nepali Sentiment (Nepal-aware)":
        # Documented on model card: Label 0 = Negative, 1 = Positive, 2 = Neutral
        manual_map = {
            "label_0": "Negative",
            "label_1": "Positive",
            "label_2": "Neutral",
        }
        if raw in manual_map:
            return manual_map[raw]
        if raw in ("positive", "negative", "neutral"):
            return raw.capitalize()
        return raw_label

    if model_name == "mBERT Multilingual (general)":
        stars = int(raw_label[0])
        if stars <= 2:
            return "Negative"
        elif stars == 3:
            return "Neutral"
        else:
            return "Positive"

    return raw_label


# =======================================================================
# 5. RUN PREDICTIONS
# =======================================================================
results = {}

for model_name, clf in pipelines.items():
    print(f"\nRunning: {model_name}")
    preds = []
    for text in df["text"]:
        raw = clf(str(text), truncation=True)[0]["label"]
        preds.append(normalize_label(model_name, raw))
    results[model_name] = preds

for model_name, preds in results.items():
    df[model_name] = preds

df.to_csv("predictions.csv", index=False)
print("\nSaved per-sentence predictions to predictions.csv")

# =======================================================================
# 6. METRICS
# =======================================================================
summary_rows = []
for model_name in results:
    acc = accuracy_score(df["sentiment"], df[model_name])
    f1 = f1_score(df["sentiment"], df[model_name], average="macro", labels=LABELS)
    summary_rows.append({"model": model_name, "accuracy": acc, "macro_f1": f1})

summary = pd.DataFrame(summary_rows).sort_values("accuracy", ascending=False)
summary.to_csv("summary_metrics.csv", index=False)
print("\n=== OVERALL RESULTS ===")
print(summary.to_string(index=False))

# =======================================================================
# 7. ACCURACY BY CODE-MIXING LEVEL
# =======================================================================
level_rows = []
for model_name in results:
    for level in ["low", "medium", "high"]:
        sub = df[df["mixing_level"] == level]
        if len(sub) == 0:
            continue
        acc = accuracy_score(sub["sentiment"], sub[model_name])
        level_rows.append({"model": model_name, "mixing_level": level, "accuracy": acc, "n": len(sub)})
level_df = pd.DataFrame(level_rows)
level_df.to_csv("accuracy_by_mixing_level.csv", index=False)
print("\n=== ACCURACY BY MIXING LEVEL ===")
print(level_df.to_string(index=False))

# =======================================================================
# 8. PLOTS
# =======================================================================
plt.figure(figsize=(9, 5))
plt.bar(summary["model"], summary["accuracy"], color="#2E5A8A")
plt.ylabel("Accuracy")
plt.title("Overall Accuracy by Model")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("fig1_overall_accuracy.png", dpi=150)
plt.close()

pivot = level_df.pivot(index="mixing_level", columns="model", values="accuracy")
pivot = pivot.reindex(["low", "medium", "high"])
pivot.plot(kind="bar", figsize=(10, 5))
plt.ylabel("Accuracy")
plt.title("Accuracy by Code-Mixing Level")
plt.xticks(rotation=0)
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("fig2_accuracy_by_mixing_level.png", dpi=150)
plt.close()

best_model = summary.iloc[0]["model"]
cm = confusion_matrix(df["sentiment"], df[best_model], labels=LABELS)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix - {best_model}")
plt.xticks(range(3), LABELS)
plt.yticks(range(3), LABELS)
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.savefig("fig3_confusion_matrix_best_model.png", dpi=150)
plt.close()

print("\nSaved: fig1_overall_accuracy.png, fig2_accuracy_by_mixing_level.png, fig3_confusion_matrix_best_model.png")
print("DONE.")