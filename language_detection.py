#model xlm-roberta-base-language-detection

from transformers import pipeline

text = [
    "What language is this?",
    "Полагаю это русский"
]

model_ckpt = "papluca/xlm-roberta-base-language-detection"
pipe = pipeline("text-classification", model=model_ckpt)
pipe(text, top_k=1, truncation=True)
