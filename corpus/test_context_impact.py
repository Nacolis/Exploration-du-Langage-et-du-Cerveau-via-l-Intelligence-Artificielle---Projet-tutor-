import torch
from transformers import GPT2Tokenizer, GPT2Model
from torch.nn.functional import cosine_similarity

def compare_evolution(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
    
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    

    base_layer = outputs.hidden_states[0]
    final_layer = outputs.hidden_states[12]
    
    sim = cosine_similarity(base_layer, final_layer, dim=2)
    return sim.mean().item()

phrase_sens = "The scientist discovered a new cure for the disease."
phrase_sens2 = "I love my girlfriend Adele, she is the best."
phrase_nimporte_quoi = "Table sky running green idea desk fast."
phrase_chaos = "ijidjfirjzi fjijfi iedie ikdqiq"
phrase_jabberwocky = "He took his vorpal sword in hand: Long time the manxome foe he sough"
phrase_nim= "Car apple roof dvd light chair gutter syringe asphalt"

score_sens = compare_evolution(phrase_sens)
score_sens2 = compare_evolution(phrase_sens2)

score_nimp = compare_evolution(phrase_nimporte_quoi)
score_chaos = compare_evolution(phrase_chaos)
score_jabberwocky = compare_evolution(phrase_jabberwocky)
score_nim = compare_evolution(phrase_nim)

print(f"Similarité Base/Finale (Phrase Logique): {score_sens:.4f}")
print(f"Similarité Base/Finale (Phrase Logique 2): {score_sens2:.4f}")
print(f"Similarité Base/Finale (Phrase Aléatoire): {score_nimp:.4f}")
print(f"Similarité Base/Finale (Phrase Nonsense): {score_chaos:.4f}")
print(f"Similarité Base/Finale (Phrase Poétique): {score_jabberwocky:.4f}")
print(f"Similarité Base/Finale (Phrase Nonsense 2): {score_nim:.4f}")
print("\nNote: Une similarité plus basse signifie que le modèle a plus transformé les vecteurs via le contexte.")
