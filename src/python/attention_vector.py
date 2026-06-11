# =========================
# 0) INSTALACIÓN
# =========================
# Si ya lo tienes instalado, esta línea la puedes comentar
#!pip install transformers torch -q

# =========================
# 1) IMPORTS
# ========================= versión de prueba para una sola frase.
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# 2) CARGAR TOKENIZER Y MODELO
# =========================
model_name = "textattack/bert-base-uncased-SST-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    output_attentions=True
)

model.eval()

# =========================
# 3) FRASE A ANALIZAR
# =========================
sentence = "The movie was surprisingly good."

# Tokenización
inputs = tokenizer(sentence, return_tensors="pt")

# Inferencia sin gradientes
with torch.no_grad():
    outputs = model(**inputs)

# Tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Atenciones
attentions = outputs.attentions

# =========================
# 4) MOSTRAR INFO BÁSICA
# =========================
print("Frase:", sentence)
print("Tokens:", tokens)
print("Número de capas:", len(attentions))
print("Número de heads por capa:", attentions[0].shape[1])
print("Shape de una capa:", attentions[0].shape)  
# debería ser: [1, 12, seq_len, seq_len]

# =========================
# 5) FUNCIÓN PARA OBTENER EL VECTOR COMPLETO
# =========================
def get_attention_vector(attentions):
    vector = []

    num_layers = len(attentions)

    for layer in range(num_layers):  # capas 0 -> 11
        # attentions[layer] tiene forma [1, 12, seq_len, seq_len]
        # quitamos la dimensión batch -> [12, seq_len, seq_len]
        layer_att = attentions[layer][0].detach().cpu().numpy()

        num_heads = layer_att.shape[0]

        for head in range(num_heads):  # heads 0 -> 11
            head_matrix = layer_att[head]  # [seq_len, seq_len]

            # Aplanado fila por fila:
            # primero toda la fila [CLS], luego la fila "the", etc.
            vector.extend(head_matrix.flatten())

    return np.array(vector)

# =========================
# 6) OBTENER VECTOR FINAL
# =========================
attention_vector = get_attention_vector(attentions)

print("\nLongitud del vector:", len(attention_vector))
print("Primeros 20 valores:")
print(attention_vector[:20])

# =========================
# 7) OPCIONAL: VER LA MATRIZ DE LA CAPA 0, HEAD 0  sirve para ver una matriz concreta: capa 0, cabeza 0. Eso te ayuda a entender qué hay dentro de una variable de atención antes de aplanarlo todo.
# =========================
head_0_layer_0 = attentions[0][0, 0].detach().cpu().numpy()

print("\nMatriz de atención de la capa 0, head 0:")
print(head_0_layer_0)

print("\nLongitud de esta head aplanada:", len(head_0_layer_0.flatten()))
print("Vector de la capa 0, head 0:")
print(head_0_layer_0.flatten())

# =========================
# 8) OPCIONAL: GUARDAR EL VECTOR EN UN CSV
# =========================
np.savetxt("attention_vector_9216.csv", attention_vector, delimiter=",")

print("\nArchivo guardado como: attention_vector_9216.csv")