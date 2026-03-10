import re
import nltk
#import spacy
#import json
from nltk.tokenize import word_tokenize

nltk.download('punkt')

with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # [^\w\s] : sélectionne tout caractère qui n'est ni une lettre/chiffre/_ (\w), ni un espace (\s)
    token = word_tokenize(text)
    print(token)


