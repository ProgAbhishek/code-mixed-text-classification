
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from preprocess import normalize_text, ID2LABEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="final")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    print("Nepali-English Code-Mixed Sentiment Demo")
    print("Type a sentence (or 'quit' to exit):\n")

    while True:
        text = input("> ").strip()
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue

        clean = normalize_text(text)
        enc = tokenizer(clean, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)[0]
            pred_id = torch.argmax(probs).item()

        print(f"  Sentiment: {ID2LABEL[pred_id]}")
        for i, p in enumerate(probs):
            print(f"    {ID2LABEL[i]}: {p.item():.3f}")
        print()


if __name__ == "__main__":
    main()
