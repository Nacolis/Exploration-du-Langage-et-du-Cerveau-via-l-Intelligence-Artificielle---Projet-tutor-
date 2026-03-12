import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Appareil de calcul utilisé : {device}")

def load_and_test_model(model_name):
    print(f"\n--- Chargement de {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        output_hidden_states=True,
        torch_dtype=torch.float16
    ).to(device)
    
    input_text = "Le chat mange la souris."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    hidden_states = outputs.hidden_states
    
    print(f"Nombre de couches récupérées : {len(hidden_states)}")
    print(f"Forme du tenseur de la dernière couche : {hidden_states[-1].shape}")
    
    return True

try:
    load_and_test_model("gpt2") 
    print("✅ GPT-2 chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur GPT-2 : {e}")

