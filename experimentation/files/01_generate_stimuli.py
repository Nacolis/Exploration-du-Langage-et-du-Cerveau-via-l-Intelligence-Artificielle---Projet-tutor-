import json
import re
import random
import unicodedata
from pathlib import Path

import spacy
from datasets import load_dataset

SEED = 42
random.seed(SEED)

MIN_WORDS = 8
MAX_WORDS = 20
N_SENTENCES = 500
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

ONSETS = [
    "bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr",
    "sc", "sk", "sl", "sm", "sn", "sp", "st", "sw", "tr", "tw",
    "b", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r",
    "s", "t", "v", "w", "z",
]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ou", "ee", "oo", "ea"]
CODAS = ["", "n", "t", "s", "ld", "nd", "st", "k", "m", "p", "r", "l", "ng"]


def make_pseudo_word(n_syllables: int = None) -> str:
    if n_syllables is None:
        n_syllables = random.choices([1, 2, 3], weights=[3, 5, 2])[0]
    word = ""
    for _ in range(n_syllables):
        word += random.choice(ONSETS) + random.choice(VOWELS) + random.choice(CODAS)
    return word

def pseudo_word_for(token) -> str:
    orig = token.text
    n_syl = max(1, round(len(orig) / 3))
    pseudo = make_pseudo_word(n_syl)

    if orig.isupper():
        return pseudo.upper()
    if orig[0].isupper():
        return pseudo.capitalize()
    return pseudo



def to_jabberwocky(doc, used_pseudos: set) -> str:
    tokens_out = []
    for token in doc:
        if token.pos_ in CONTENT_POS and not token.is_punct:
            for _ in range(20):
                pseudo = pseudo_word_for(token)
                if pseudo not in used_pseudos:
                    used_pseudos.add(pseudo)
                    break
            tokens_out.append(pseudo)
        else:
            tokens_out.append(token.text)
    return "".join(
        t + doc[i].whitespace_
        for i, t in enumerate(tokens_out)
    ).strip()



def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(u'[\u2018\u2019\u0060]', "'", text)
    text = re.sub(u'[\u201c\u201d]', '"', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def is_valid(text: str) -> bool:
    words = text.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False
    if re.search(r"\d", text):
        return False
    if re.search(r"http|www|<|>|=", text):
        return False
    if not text[-1] in ".!?":
        return False
    if "(" in text or ")" in text:
        return False
    return True


def main():
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    raw_sentences = []
    for item in dataset:
        text = item["text"].strip()
        if not text:
            continue
        parts = re.split(r"(?<=[.!?])\s+", text)
        for p in parts:
            p = clean(p)
            if is_valid(p):
                raw_sentences.append(p)

    random.shuffle(raw_sentences)
    print(f"Found {len(raw_sentences)} candidate sentences after filtering.")

    if len(raw_sentences) < N_SENTENCES:
        raise ValueError(
            f"Only {len(raw_sentences)} valid sentences found, need {N_SENTENCES}. "
            "Consider relaxing filters."
        )

    selected = raw_sentences[:N_SENTENCES]

    print("Parsing and generating Jabberwocky variants...")
    used_pseudos: set = set()
    pairs = []

    docs = list(nlp.pipe(selected, batch_size=64))
    for i, doc in enumerate(docs):
        semantic = doc.text
        jabber = to_jabberwocky(doc, used_pseudos)

        content_count = sum(1 for t in doc if t.pos_ in CONTENT_POS and not t.is_punct)
        total_count = sum(1 for t in doc if not t.is_punct and not t.is_space)

        pairs.append({
            "id": i,
            "semantic": semantic,
            "jabberwocky": jabber,
            "n_words": len(semantic.split()),
            "content_ratio": round(content_count / total_count, 3) if total_count else 0,
        })

    print("\n── Sample pairs ──────────────────────────────")
    for p in pairs[:5]:
        print(f"[{p['id']}] SEM: {p['semantic']}")
        print(f"      JAB: {p['jabberwocky']}")
        print()

    out_path = OUT_DIR / "stimuli.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(pairs)} pairs → {out_path}")
    print(f"Word length distribution: "
          f"min={min(p['n_words'] for p in pairs)}, "
          f"max={max(p['n_words'] for p in pairs)}, "
          f"mean={sum(p['n_words'] for p in pairs)/len(pairs):.1f}")


if __name__ == "__main__":
    main()