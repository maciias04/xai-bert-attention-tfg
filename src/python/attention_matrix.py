import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# 1) CARGAR TOKENIZER Y MODELO
# =========================
model_name = "textattack/bert-base-uncased-SST-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    output_attentions=True
)
model.eval()

# =========================
# 2) FRASES CANDIDATAS
# Todas con 5 palabras + punto final
# Objetivo: [CLS] + 5 tokens + . + [SEP] = 8 tokens
# =========================
positive_sentences = [
    "The movie was very good.",
    "This film was really nice.",
    "I found the story charming.",
    "The acting felt quite natural.",
    "This movie looked very beautiful.",
    "The ending was truly satisfying.",
    "I enjoyed this film today.",
    "The plot felt warm throughout.",
    "This was a lovely movie.",
    "The cast gave strong performances.",
    "I liked the final scene.",
    "This film felt deeply moving.",
    "The soundtrack was very pleasant.",
    "The script was smart overall.",
    "I found it quite enjoyable.",
    "The movie felt fresh today.",
    "This story was very warm.",
    "The pacing worked very well.",
    "I loved this movie completely.",
    "The dialogue felt sharp throughout."
]

negative_sentences = [
    "The movie was very bad.",
    "This film was really awful.",
    "I found the story boring.",
    "The acting felt quite wooden.",
    "This movie looked very cheap.",
    "The ending was truly annoying.",
    "I disliked this film today.",
    "The plot felt weak throughout.",
    "This was a dreadful movie.",
    "The cast gave poor performances.",
    "I hated the final scene.",
    "This film felt deeply empty.",
    "The soundtrack was very unpleasant.",
    "The script was dumb overall.",
    "I found it quite painful.",
    "The movie felt stale today.",
    "This story was pointless indeed.",
    "The pacing worked very poorly.",
    "I hated this movie completely.",
    "The dialogue felt flat throughout."
]

all_sentences = positive_sentences + negative_sentences
labels = ["positive"] * len(positive_sentences) + ["negative"] * len(negative_sentences)

# =========================
# 3) COMPROBAR QUE TODAS PRODUCEN 8 TOKENS
# Si alguna no cumple, el programa se detiene
# =========================
def check_sentence_tokens(sentences, tokenizer):
    for s in sentences:
        inputs = tokenizer(s, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        print(f"{s} -> {tokens} -> {len(tokens)} tokens")
        if len(tokens) != 8:
            raise ValueError(
                f'La frase "{s}" no produce 8 tokens, sino {len(tokens)}.'
            )

print("Verificando frases positivas...")
check_sentence_tokens(positive_sentences, tokenizer)

print("\nVerificando frases negativas...")
check_sentence_tokens(negative_sentences, tokenizer)

print("\nTodas las frases producen exactamente 8 tokens.")

# =========================
# 4) FUNCIÓN PARA OBTENER EL VECTOR DE 9216
# Orden:
# capa 0 -> head 0 -> fila 0 completa, fila 1 completa...
# ...
# capa 11 -> head 11
# =========================
def get_attention_vector(attentions):
    vector = []

    for layer in range(len(attentions)):  # 12 capas
        layer_att = attentions[layer][0].detach().cpu().numpy()  # [12, 8, 8]

        for head in range(layer_att.shape[0]):  # 12 heads
            head_matrix = layer_att[head]  # [8, 8]
            vector.extend(head_matrix.flatten())

    return np.array(vector, dtype=np.float32)

# =========================
# 5) PROCESAR UNA FRASE
# =========================
def process_sentence(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attentions = outputs.attentions
    probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy()

    attention_vector = get_attention_vector(attentions)

    return tokens, attention_vector, probs

# =========================
# 6) CONSTRUIR MATRIZ FINAL 40 x 9216
# =========================
matrix_40x9216 = []
tokens_per_sentence = []
prob_negative = []
prob_positive = []
predicted_label = []

label_names = ["negative", "positive"]

for i, sentence in enumerate(all_sentences):
    tokens, vector, probs = process_sentence(sentence, tokenizer, model)

    if len(tokens) != 8:
        raise ValueError(
            f'La frase "{sentence}" no tiene 8 tokens al procesarla.'
        )

    if len(vector) != 9216:
        raise ValueError(
            f'La frase "{sentence}" no produce 9216 valores, sino {len(vector)}.'
        )

    matrix_40x9216.append(vector)
    tokens_per_sentence.append(tokens)
    prob_negative.append(float(probs[0]))
    prob_positive.append(float(probs[1]))
    predicted_label.append(label_names[int(np.argmax(probs))])

    print(f"\nFrase {i+1:02d}/40")
    print("Texto:", sentence)
    print("Tokens:", tokens)
    print("Longitud vector:", len(vector))
    print("Predicción:", predicted_label[-1])
    print("Probabilidades:", {"negative": float(probs[0]), "positive": float(probs[1])})

matrix_40x9216 = np.array(matrix_40x9216, dtype=np.float32)

print("\nShape final de la matriz:", matrix_40x9216.shape)  # (40, 9216)

# =========================
# 7) GUARDAR MATRIZ PRINCIPAL
# =========================
feature_columns = [f"att_{i}" for i in range(9216)]

df_matrix = pd.DataFrame(matrix_40x9216, columns=feature_columns)
df_matrix.insert(0, "predicted_label", predicted_label)
df_matrix.insert(0, "prob_positive", prob_positive)
df_matrix.insert(0, "prob_negative", prob_negative)
df_matrix.insert(0, "true_label", labels)
df_matrix.insert(0, "sentence", all_sentences)

df_matrix.to_csv("attention_matrix_40x9216.csv", index=False)

print("\nArchivo guardado: attention_matrix_40x9216.csv")

# =========================
# 8) GUARDAR TOKENS DE CADA FRASE
# =========================
df_tokens = pd.DataFrame({
    "sentence": all_sentences,
    "true_label": labels,
    "predicted_label": predicted_label,
    "tokens": [" ".join(toks) for toks in tokens_per_sentence]
})

df_tokens.to_csv("attention_tokens_40_sentences.csv", index=False)

print("Archivo guardado: attention_tokens_40_sentences.csv")