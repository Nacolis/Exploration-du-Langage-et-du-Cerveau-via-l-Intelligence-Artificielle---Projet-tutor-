import spacy
import torch
from transformers import GPT2Tokenizer, GPT2Model

def test_gpt2_spacy_alignment(text):
    # 1. Initialisation de spaCy et GPT-2
    nlp = spacy.load("en_core_web_sm")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)

    doc = nlp(text)
    spacy_tokens = [token.text for token in doc]
    print(f"Tokens spaCy : {spacy_tokens}")


    inputs = tokenizer(text, return_tensors="pt")
    gpt2_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(f"Tokens GPT-2 : {gpt2_tokens}")

    # 4. Extraction des hidden states
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states
    print(f"Nombre de couches extraites : {len(hidden_states)}")

    # 5. Alignement (Logique simple de reconstruction)
    alignment = []
    gpt2_idx = 0
    
    for word in spacy_tokens:
        current_word_tokens = []
        reconstructed = ""
        
        while gpt2_idx < len(gpt2_tokens):
            token = gpt2_tokens[gpt2_idx]
            clean_token = token.replace('Ġ', '')
            
            reconstructed += clean_token
            current_word_tokens.append((token, gpt2_idx))
            gpt2_idx += 1
            
            if reconstructed.lower() == word.lower() or (reconstructed == "" and clean_token == ""):
                break
            
            if len(reconstructed) > len(word) and word.lower() in reconstructed.lower():
                 break

        alignment.append({
            "word": word,
            "gpt2_subwords": current_word_tokens
        })

    return alignment, hidden_states

def main():
    sentence = "Theo is a beautiful cat."
    alignment, states = test_gpt2_spacy_alignment(sentence)

    print("\n--- Alignement ---")
    for entry in alignment:
        subwords = [t[0] for t in entry['gpt2_subwords']]
        print(f"Mot: {entry['word']:<10} | Sous-mots GPT-2: {subwords}")

    print(f"\nTaille des hidden states (Couche 12) : {states[12].shape}")

if __name__ == "__main__":
    main()
