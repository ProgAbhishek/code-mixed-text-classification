
import re
import pandas as pd

LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")  # e.g. "sooo" -> "soo"

# Common romanized-Nepali spelling variants -> canonical form.
# Extend this dict as you spot more variants in your data (EDA step below).
SPELLING_MAP = {
    "vayo": "bhayo", "vo": "bho", "vaneko": "bhaneko",
    "xa": "cha", "xaina": "chaina", "xu": "chu",
    "hunxa": "huncha", "garexu": "garcha",
}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = REPEAT_CHAR_RE.sub(r"\1\1", text)  # cap repeats at 2 chars
    text = text.lower().strip()

    tokens = text.split()
    tokens = [SPELLING_MAP.get(tok, tok) for tok in tokens]
    text = " ".join(tokens)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_split(path: str) -> pd.DataFrame:
    """Load a TSV with columns: id, text, sentiment."""
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["text", "sentiment"])
    df["text_clean"] = df["text"].apply(normalize_text)
    df = df[df["text_clean"].str.len() > 0]
    df["label"] = df["sentiment"].map(LABEL2ID)
    if df["label"].isna().any():
        bad = df[df["label"].isna()]["sentiment"].unique()
        raise ValueError(f"Unmapped sentiment labels found: {bad}")
    df["label"] = df["label"].astype(int)
    return df


if __name__ == "__main__":
    import sys
    df = load_split(sys.argv[1] if len(sys.argv) > 1 else "data/val.tsv")
    print(df[["text", "text_clean", "sentiment", "label"]].head(10))
    print("\nLabel distribution:\n", df["sentiment"].value_counts())
