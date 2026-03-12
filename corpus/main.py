import spacy
from datasets import load_dataset
import json
from transformers import GPT2Tokenizer, GPT2Model
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)

def preprocess_corpus(limit_chars=50000):
    print("Loading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split='test', streaming=True)
    
    nlp = spacy.load("en_core_web_sm")


    raw_text = ""
    for entry in dataset:
        if len(raw_text) > limit_chars:
            break
        raw_text += entry['text'] + "\n"

    paragraphs_raw = [p.strip() for p in raw_text.split('\n\n') if p.strip()]
    
    structured_corpus = []
    global_token_pos = 0

    print(f"Processing {len(paragraphs_raw)} paragraphs...")

    for p_idx, p_text in enumerate(paragraphs_raw):
        doc = nlp(p_text)
        
        for s_idx, sent in enumerate(doc.sents):
            for t_idx, token in enumerate(sent):
                if token.is_space:
                    continue

                token_info = {
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "para_idx": p_idx,
                    "sent_idx": s_idx,
                    "pos_in_sent": t_idx,
                    "global_pos": global_token_pos
                }
                
                structured_corpus.append(token_info)
                global_token_pos += 1

    return structured_corpus



def main():
    # --- Exécution ---
    corpus_annoté = preprocess_corpus(limit_chars=10000)

    print("\nExemple de structure pour un token :")

    print(json.dumps(corpus_annoté[50]))
    print(json.dumps(corpus_annoté[100]))


    nouns_p2 = [t['text'] for t in corpus_annoté if t['para_idx'] == 2 and t['pos'] == 'NOUN']
    print(f"\nNoms trouvés dans le paragraphe 2 : {nouns_p2[:10]}...")


if __name__ == "__main__":
    main()
