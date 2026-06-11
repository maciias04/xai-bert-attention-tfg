import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
# ===============================
# 1) Cargar importancia global SHAP
# ===============================

df_shap = pd.read_csv("shap_global_token_importance.csv")

# Diccionario: token -> importancia media |SHAP|
shap_dict = dict(zip(df_shap["token_clean"], df_shap["mean_abs_shap"]))


# ===============================
# 2) Frases usadas en el experimento
# ===============================

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
    "This story was very dull.",
    "The pacing worked very poorly.",
    "I hated this movie completely.",
    "The dialogue felt flat throughout."
]

sentences = positive_sentences + negative_sentences


# ===============================
# 3) Función simple de tokenización
# ===============================

def clean_sentence(sentence):
    """
    Convierte:
    'The movie was very good.'
    en:
    ['the', 'movie', 'was', 'very', 'good', '.']
    """
    sentence = sentence.lower().strip()
    sentence = sentence.replace(".", " .")
    tokens = sentence.split()
    return tokens


# ===============================
# 4) Acumular importancia por posición
# ===============================

# Posiciones BERT:
# 1 -> [CLS]
# 2 -> palabra 1
# 3 -> palabra 2
# 4 -> palabra 3
# 5 -> palabra 4
# 6 -> palabra 5
# 7 -> .
# 8 -> [SEP]

importance_by_position = np.zeros(8)
count_by_position = np.zeros(8)

for sentence in sentences:
    tokens = clean_sentence(sentence)

    if len(tokens) != 6:
        raise ValueError(f"La frase no tiene 5 palabras + punto: {sentence} -> {tokens}")

    # [CLS] posición 1 -> no tiene SHAP directo
    # [SEP] posición 8 -> no tiene SHAP directo

    for i, tok in enumerate(tokens): #Recorre los tokens de la frase.
        # i = 0 corresponde a posición 2 en BERT
        bert_pos = i + 1

        shap_value = shap_dict.get(tok, 0.0) #Busca la importancia SHAP global del token actual.

        importance_by_position[bert_pos] += shap_value #Suma la importancia SHAP de ese token a su posición correspondiente. Si good aparece como quinta palabra, se suma a la posición 5 del vector.
        count_by_position[bert_pos] += 1 #Aumenta el contador de esa posición,Sirve para luego calcular la media. Por ejemplo, si en la posición 5 han aparecido 40 palabras, se dividirá entre 40.


# Media por posición
mean_importance_by_position = np.zeros(8) #Crea un vector de 8 posiciones para guardar la importancia media por posición.

for i in range(8): #para calculara la media final de cada posicion de token
    if count_by_position[i] > 0: #Esto evita dividir entre cero.
        mean_importance_by_position[i] = importance_by_position[i] / count_by_position[i] #Calcula la media de importancia en esa posición.
    else:
        mean_importance_by_position[i] = 0.0


# Normalizar a porcentaje
importance_pct = 100 * mean_importance_by_position / mean_importance_by_position.sum()
#La suma final de todos los porcentajes será 100 %.

# ===============================
# 5) Guardar CSV comparable
# ===============================

token_labels = ["[CLS]", "Token 2", "Token 3", "Token 4", "Token 5", "Token 6", ".", "[SEP]"]

df_out = pd.DataFrame({
    "Token": token_labels,
    "Importancia": importance_pct
})

df_out.to_csv("shap_importancia_tokens.csv", index=False)

print(df_out)
print("\nArchivo guardado: shap_importancia_tokens.csv")

# ============================================================
# 5) Generar imagen
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(token_labels, importance_pct)

plt.xlabel("Posición de token", fontsize=14, fontweight="bold")
plt.ylabel("Importancia relativa SHAP (%)", fontsize=14, fontweight="bold")
plt.title("SHAP: importancia relativa por posición de token", fontsize=16, fontweight="bold")

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("shap_importancia_tokens.png", dpi=300)
plt.show()

print("Imagen guardada: shap_importancia_tokens.png")
print("Tabla guardada: shap_importancia_tokens.csv")