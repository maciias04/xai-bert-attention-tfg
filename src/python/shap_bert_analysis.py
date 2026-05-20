# ============================================================
# SHAP SOBRE BERT PARA CLASIFICACIÓN DE SENTIMIENTO
# Autor: Adrián Macias Caballero
#
# Objetivo:
# - Aplicar SHAP al modelo textattack/bert-base-uncased-SST-2
# - Usar las mismas 40 frases del análisis de atención
# - Obtener importancia de tokens para cada predicción
# - Guardar resultados en CSV, HTML y PNG
# ============================================================

import os
import re
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# 0) CONFIGURACIÓN GENERAL
# ============================================================

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "textattack/bert-base-uncased-SST-2"

# Longitud usada en tu TFG:
# [CLS] + 5 palabras + "." + [SEP] = 8 tokens
MAX_LENGTH = 8

CLASS_NAMES = ["negative", "positive"]

OUTPUT_DIR = "shap_results"
HTML_DIR = os.path.join(OUTPUT_DIR, "html")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# 1) CARGAR TOKENIZER Y MODELO
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Dispositivo usado:", device)
print("Cargando modelo:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()


# ============================================================
# 2) FRASES DEL EXPERIMENTO
# ============================================================

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
true_labels = ["positive"] * len(positive_sentences) + ["negative"] * len(negative_sentences)


# ============================================================
# 3) FUNCIONES AUXILIARES
# ============================================================

def clean_token(token: str) -> str:
    """
    Limpia tokens o segmentos de texto para hacer rankings globales.
    """
    token = str(token)
    token = token.replace("##", "")
    token = token.strip()
    token = token.lower()
    token = re.sub(r"\s+", " ", token)
    return token


def check_sentence_tokens(sentences_list, tokenizer_obj):
    """
    Comprueba que cada frase produce 8 tokens:
    [CLS] + 5 tokens + . + [SEP]
    """
    token_rows = []

    for idx, sentence in enumerate(sentences_list):
        inputs = tokenizer_obj(sentence, return_tensors="pt")
        tokens = tokenizer_obj.convert_ids_to_tokens(inputs["input_ids"][0])

        token_rows.append({
            "sentence_id": idx + 1,
            "sentence": sentence,
            "tokens": " ".join(tokens),
            "n_tokens": len(tokens)
        })

        print(f"{idx+1:02d}. {sentence}")
        print("    Tokens:", tokens)
        print("    Nº tokens:", len(tokens))

        if len(tokens) != MAX_LENGTH:
            raise ValueError(
                f'La frase "{sentence}" produce {len(tokens)} tokens, '
                f'pero se esperaban {MAX_LENGTH}.'
            )

    return pd.DataFrame(token_rows)


def predict_proba(texts):
    """
    Función que SHAP va a explicar.
    Recibe una lista de textos y devuelve probabilidades:
    columna 0 -> negative
    columna 1 -> positive
    """

    # SHAP puede pasar textos como numpy array.
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()

    # Asegurar lista de strings.
    if isinstance(texts, str):
        texts = [texts]

    texts = [str(t) for t in texts]

    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    return probs.detach().cpu().numpy()


def get_predictions(sentences_list):
    """
    Obtiene predicciones y probabilidades del modelo para cada frase.
    """
    probs = predict_proba(sentences_list)

    predicted_ids = np.argmax(probs, axis=1)
    predicted_labels = [CLASS_NAMES[i] for i in predicted_ids]

    df_pred = pd.DataFrame({
        "sentence_id": np.arange(1, len(sentences_list) + 1),
        "sentence": sentences_list,
        "true_label": true_labels,
        "predicted_label": predicted_labels,
        "prob_negative": probs[:, 0],
        "prob_positive": probs[:, 1]
    })

    return df_pred


def extract_values_for_class(shap_values, sample_idx, class_idx):
    """
    Extrae los valores SHAP de una frase para una clase concreta.

    En SHAP para texto, normalmente:
    shap_values.values[sample_idx] -> matriz [tokens, clases]

    Esta función lo hace robusto por si cambia el formato.
    """
    values = shap_values.values[sample_idx]

    if values.ndim == 2:
        return values[:, class_idx]
    elif values.ndim == 1:
        return values
    else:
        raise ValueError(f"Formato inesperado de SHAP values: {values.shape}")


def extract_tokens_from_shap(shap_values, sample_idx):
    """
    Extrae los tokens o segmentos usados por SHAP.
    """
    data = shap_values.data[sample_idx]

    if isinstance(data, str):
        return [data]

    return [str(x) for x in data]


def save_shap_text_html(shap_values, sample_idx, class_idx, filename):
    """
    Guarda la visualización textual de SHAP en HTML.
    """
    try:
        explanation = shap_values[sample_idx, :, class_idx]
        html_obj = shap.plots.text(explanation, display=False)
        shap.save_html(filename, html_obj)
    except Exception as e:
        print(f"No se pudo guardar HTML para muestra {sample_idx+1}: {e}")


def plot_token_bar(tokens, values, title, filename, top_k=12):
    """
    Guarda una gráfica de barras horizontales con los tokens más importantes.
    """
    tokens = np.array(tokens)
    values = np.array(values, dtype=float)

    # Filtrar tokens vacíos
    valid = np.array([str(t).strip() != "" for t in tokens])
    tokens = tokens[valid]
    values = values[valid]

    if len(tokens) == 0:
        return

    # Ordenar por importancia absoluta
    order = np.argsort(np.abs(values))[::-1]
    order = order[:min(top_k, len(order))]

    tokens_top = tokens[order]
    values_top = values[order]

    # Para que el más importante quede arriba
    tokens_top = tokens_top[::-1]
    values_top = values_top[::-1]

    plt.figure(figsize=(9, 4.8))
    plt.barh(tokens_top, values_top)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Valor SHAP")
    plt.ylabel("Token")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# 4) COMPROBAR TOKENS Y PREDICCIONES
# ============================================================

print("\n============================================================")
print("COMPROBANDO TOKENIZACIÓN")
print("============================================================")

df_tokens = check_sentence_tokens(sentences, tokenizer)
df_tokens.to_csv(os.path.join(DATA_DIR, "tokens_40_sentences.csv"), index=False)

print("\n============================================================")
print("OBTENIENDO PREDICCIONES")
print("============================================================")

df_predictions = get_predictions(sentences)
df_predictions.to_csv(os.path.join(DATA_DIR, "bert_predictions_40_sentences.csv"), index=False)

print(df_predictions)


# ============================================================
# 5) CREAR EXPLICADOR SHAP
# ============================================================

print("\n============================================================")
print("CREANDO EXPLICADOR SHAP")
print("============================================================")

# Masker de texto. SHAP irá ocultando partes del texto y midiendo
# cómo cambia la predicción del modelo.
masker = shap.maskers.Text(tokenizer)

explainer = shap.Explainer(
    predict_proba,
    masker,
    output_names=CLASS_NAMES
)


# ============================================================
# 6) CALCULAR SHAP
# ============================================================

print("\n============================================================")
print("CALCULANDO SHAP PARA LAS 40 FRASES")
print("============================================================")

# Para frases cortas, esto suele ser suficiente.
# Si tarda demasiado, puedes bajar max_evals.
shap_values = explainer(
    sentences,
    max_evals=500,
    batch_size=8
)

print("SHAP calculado correctamente.")


# ============================================================
# 7) GUARDAR RESULTADOS POR FRASE
# ============================================================

print("\n============================================================")
print("GUARDANDO RESULTADOS SHAP")
print("============================================================")

rows = []
summary_rows = []

for i, sentence in enumerate(sentences):
    true_label = df_predictions.loc[i, "true_label"]
    pred_label = df_predictions.loc[i, "predicted_label"]
    prob_neg = df_predictions.loc[i, "prob_negative"]
    prob_pos = df_predictions.loc[i, "prob_positive"]

    pred_class_idx = CLASS_NAMES.index(pred_label)

    shap_tokens = extract_tokens_from_shap(shap_values, i)
    shap_vals_pred = extract_values_for_class(shap_values, i, pred_class_idx)

    # Guardar HTML de SHAP para la clase predicha
    html_name = os.path.join(
        HTML_DIR,
        f"sentence_{i+1:02d}_shap_{pred_label}.html"
    )
    save_shap_text_html(shap_values, i, pred_class_idx, html_name)

    # Guardar PNG tipo barras para la clase predicha
    plot_name = os.path.join(
        PLOTS_DIR,
        f"sentence_{i+1:02d}_bar_{pred_label}.png"
    )

    plot_token_bar(
        shap_tokens,
        shap_vals_pred,
        title=f"SHAP frase {i+1:02d} - clase explicada: {pred_label}",
        filename=plot_name,
        top_k=12
    )

    # Guardar valores de cada token
    for pos, (tok, val) in enumerate(zip(shap_tokens, shap_vals_pred)):
        rows.append({
            "sentence_id": i + 1,
            "sentence": sentence,
            "true_label": true_label,
            "predicted_label": pred_label,
            "prob_negative": prob_neg,
            "prob_positive": prob_pos,
            "explained_class": pred_label,
            "token_position": pos,
            "token_or_segment": tok,
            "token_clean": clean_token(tok),
            "shap_value": float(val),
            "abs_shap_value": float(abs(val))
        })

    # Top tokens de la frase
    abs_vals = np.abs(shap_vals_pred)
    order = np.argsort(abs_vals)[::-1]

    top_tokens = []
    top_values = []

    for idx_tok in order[:5]:
        if str(shap_tokens[idx_tok]).strip() == "":
            continue
        top_tokens.append(str(shap_tokens[idx_tok]))
        top_values.append(float(shap_vals_pred[idx_tok]))

    summary_rows.append({
        "sentence_id": i + 1,
        "sentence": sentence,
        "true_label": true_label,
        "predicted_label": pred_label,
        "prob_negative": prob_neg,
        "prob_positive": prob_pos,
        "top_tokens": " | ".join(top_tokens),
        "top_shap_values": " | ".join([f"{v:.6f}" for v in top_values])
    })


df_shap_long = pd.DataFrame(rows)
df_shap_summary = pd.DataFrame(summary_rows)

df_shap_long.to_csv(
    os.path.join(DATA_DIR, "shap_values_by_token.csv"),
    index=False
)

df_shap_summary.to_csv(
    os.path.join(DATA_DIR, "shap_summary_by_sentence.csv"),
    index=False
)

print("Archivo guardado:", os.path.join(DATA_DIR, "shap_values_by_token.csv"))
print("Archivo guardado:", os.path.join(DATA_DIR, "shap_summary_by_sentence.csv"))


# ============================================================
# 8) IMPORTANCIA GLOBAL DE TOKENS
# ============================================================

print("\n============================================================")
print("CALCULANDO IMPORTANCIA GLOBAL DE TOKENS")
print("============================================================")

df_valid = df_shap_long[df_shap_long["token_clean"] != ""].copy()

df_global = (
    df_valid
    .groupby("token_clean")
    .agg(
        mean_abs_shap=("abs_shap_value", "mean"),
        mean_shap=("shap_value", "mean"),
        max_abs_shap=("abs_shap_value", "max"),
        count=("token_clean", "count")
    )
    .reset_index()
    .sort_values("mean_abs_shap", ascending=False)
)

df_global.to_csv(
    os.path.join(DATA_DIR, "shap_global_token_importance.csv"),
    index=False
)

print(df_global.head(20))


# ============================================================
# 9) GRÁFICA GLOBAL DE TOKENS MÁS IMPORTANTES
# ============================================================

top_global = df_global.head(20).copy()
top_global = top_global.iloc[::-1]

plt.figure(figsize=(9, 6))
plt.barh(top_global["token_clean"], top_global["mean_abs_shap"])
plt.xlabel("Importancia media |SHAP|")
plt.ylabel("Token")
plt.title("Tokens más importantes según SHAP")
plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR, "global_top_tokens_shap.png"),
    dpi=300
)
plt.close()


# ============================================================
# 10) IMPORTANCIA MEDIA POR CLASE REAL
# ============================================================

df_by_true_class = (
    df_valid
    .groupby(["true_label", "token_clean"])
    .agg(
        mean_abs_shap=("abs_shap_value", "mean"),
        mean_shap=("shap_value", "mean"),
        count=("token_clean", "count")
    )
    .reset_index()
    .sort_values(["true_label", "mean_abs_shap"], ascending=[True, False])
)

df_by_true_class.to_csv(
    os.path.join(DATA_DIR, "shap_token_importance_by_true_class.csv"),
    index=False
)

for label in ["positive", "negative"]:
    df_label = df_by_true_class[df_by_true_class["true_label"] == label].head(15)
    df_label = df_label.iloc[::-1]

    plt.figure(figsize=(9, 5))
    plt.barh(df_label["token_clean"], df_label["mean_abs_shap"])
    plt.xlabel("Importancia media |SHAP|")
    plt.ylabel("Token")
    plt.title(f"Tokens más importantes según SHAP - frases {label}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"top_tokens_shap_true_{label}.png"),
        dpi=300
    )
    plt.close()


# ============================================================
# 11) GUARDAR CONFIGURACIÓN DEL EXPERIMENTO
# ============================================================

config = {
    "model_name": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "class_names": CLASS_NAMES,
    "n_sentences": len(sentences),
    "n_positive": len(positive_sentences),
    "n_negative": len(negative_sentences),
    "device": str(device),
    "output_dir": OUTPUT_DIR
}

with open(os.path.join(OUTPUT_DIR, "shap_experiment_config.json"), "w") as f:
    json.dump(config, f, indent=4)


# ============================================================
# 12) MENSAJE FINAL
# ============================================================

print("\n============================================================")
print("ANÁLISIS SHAP FINALIZADO")
print("============================================================")
print("Resultados guardados en:", OUTPUT_DIR)
print("\nArchivos principales:")
print("- data/bert_predictions_40_sentences.csv")
print("- data/shap_values_by_token.csv")
print("- data/shap_summary_by_sentence.csv")
print("- data/shap_global_token_importance.csv")
print("- plots/global_top_tokens_shap.png")
print("- html/sentence_XX_shap_positive.html o negative.html")