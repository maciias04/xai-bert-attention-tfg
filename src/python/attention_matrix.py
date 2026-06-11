import numpy as np #manejo de vectores y matrices numericas
import pandas as pd #para crear tablas y guardarlas como .csv
import torch #libreria sobre la que trabaja BERT , son los tensores.
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
#autotokenizer convierte frases en tokens compatibles con BERT
#automodelforsequenceclassification carga un modelo Bert para la clasificacion de texto

# =========================
# 1) CARGAR TOKENIZER Y MODELO
# =========================
model_name = "textattack/bert-base-uncased-SST-2" #define el modelo de bert a utilizar

tokenizer = AutoTokenizer.from_pretrained(model_name) #Carga el tokenizador asociado a ese modelo.Convierte la frase en tokens
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    output_attentions=True #Esto le dice al modelo que, además de devolver la predicción, también devuelva las matrices de atención internas.
)#Carga el modelo BERT de clasificación.
model.eval() #modo evaluacion,el modelo no se va a entrenar, solo se va a usar para predecir y extraer información interna.

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
    "This story was very dull.",
    "The pacing worked very poorly.",
    "I hated this movie completely.",
    "The dialogue felt flat throughout."
]

all_sentences = positive_sentences + negative_sentences
labels = ["positive"] * len(positive_sentences) + ["negative"] * len(negative_sentences) #Crea las etiquetas reales de las frases.

# =========================
# 3) COMPROBAR QUE TODAS PRODUCEN 8 TOKENS
# Si alguna no cumple, el programa se detiene
# =========================
def check_sentence_tokens(sentences, tokenizer):
    for s in sentences:
        inputs = tokenizer(s, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) #Convierte los identificadores numéricos de BERT en tokens legibles.
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
def get_attention_vector(attentions): #función que convierte todas las matrices de atención de una frase en un único vector.
    vector = []

    for layer in range(len(attentions)):  # 12 capas
        layer_att = attentions[layer][0].detach().cpu().numpy()  # [12, 8, 8]

        for head in range(layer_att.shape[0]):  # 12 heads
            head_matrix = layer_att[head]  # [8, 8]
            vector.extend(head_matrix.flatten()) #Convierte la matriz 8 × 8 en un vector de 64 valores y los añade al vector total.

    return np.array(vector, dtype=np.float32)

# =========================
# 5) PROCESAR UNA FRASE
# =========================
def process_sentence(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt")#Tokeniza la frase y la convierte en entrada para PyTorch.

    with torch.no_grad(): #Desactiva el cálculo de gradientes pq no se entrena, solo se evalua.
        outputs = model(**inputs) #Pasa la frase por BERT.

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) #Obtiene los tokens legibles de la frase.
    attentions = outputs.attentions #Extrae las atenciones internas del modelo.
    probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy() #Convierte los logits del modelo en probabilidades con softmax.

    attention_vector = get_attention_vector(attentions) #Convierte todas las matrices de atención de esa frase en un vector de 9216 valores.

    return tokens, attention_vector, probs

# =========================
# 6) CONSTRUIR MATRIZ FINAL 40 x 9216
# =========================
matrix_40x9216 = [] #Lista donde se guardará el vector de atención de cada frase.
tokens_per_sentence = [] #Lista donde se guardan los tokens de cada frase.
prob_negative = [] #Lista donde se guardan las probabilidades de clase negativa.
prob_positive = [] #Lista donde se guardan las probabilidades de clase positiva.
predicted_label = [] #Lista donde se guarda la clase predicha por BERT.

label_names = ["negative", "positive"] #Define el nombre de las clases.

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

    matrix_40x9216.append(vector) #Añade el vector de la frase a la matriz completa.
    tokens_per_sentence.append(tokens) #Guarda los tokens de la frase.
    prob_negative.append(float(probs[0])) #Guarda la probabilidad de clase negativa.
    prob_positive.append(float(probs[1])) #Guarda la probabilidad de clase positiva.
    predicted_label.append(label_names[int(np.argmax(probs))]) #Calcula la clase predicha.

    print(f"\nFrase {i+1:02d}/40") #Imprime el número de frase
    print("Texto:", sentence) #imprime la frase
    print("Tokens:", tokens) #imprime los tokens de la frase
    print("Longitud vector:", len(vector)) #longitud del vector de la frase que es 9216
    print("Predicción:", predicted_label[-1]) #Imprime la última predicción guardada.
    print("Probabilidades:", {"negative": float(probs[0]), "positive": float(probs[1])}) #Imprime las probabilidades de clase negativa y positiva.

matrix_40x9216 = np.array(matrix_40x9216, dtype=np.float32) #Convierte la lista de vectores en una matriz NumPy.

print("\nShape final de la matriz:", matrix_40x9216.shape)  # (40, 9216)

# =========================
# 7) GUARDAR MATRIZ PRINCIPAL
# =========================
feature_columns = [f"att_{i}" for i in range(9216)] #Crea los nombres de las 9216 columnas de atención.

df_matrix = pd.DataFrame(matrix_40x9216, columns=feature_columns) #Crea una tabla de Pandas con la matriz de atención.
df_matrix.insert(0, "predicted_label", predicted_label) #inserta estas columnas al inicio de la matriz
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