%% =========================================================
% COMPARACIÓN DE TOKENS REALES:
% SHAP vs oMEDA vs PLS-DA vs ASCA
% ==========================================================

clear;
close all;
clc;

%% =========================================================
% 1) CARGAR IMPORTANCIAS POR POSICIÓN
% ==========================================================

T_omeda = readtable('omeda_importancia_tokens.csv');
T_plsda = readtable('plsda_importancia_tokens.csv');
T_asca  = readtable('asca_importancia_tokens.csv');
T_shap  = readtable('shap_global_token_importance.csv');

% Importancia combinada por posición para métodos basados en atención
omeda_pos = T_omeda.Combinada(:);
plsda_pos = T_plsda.Combinada(:);
asca_pos  = T_asca.Combinada(:);

% SHAP global por token real
shap_tokens = lower(string(T_shap.token_clean));
shap_values = T_shap.mean_abs_shap(:);

%% =========================================================
% 2) FRASES DEL EXPERIMENTO
% ==========================================================

positive_sentences = [
    "The movie was very good."
    "This film was really nice."
    "I found the story charming."
    "The acting felt quite natural."
    "This movie looked very beautiful."
    "The ending was truly satisfying."
    "I enjoyed this film today."
    "The plot felt warm throughout."
    "This was a lovely movie."
    "The cast gave strong performances."
    "I liked the final scene."
    "This film felt deeply moving."
    "The soundtrack was very pleasant."
    "The script was smart overall."
    "I found it quite enjoyable."
    "The movie felt fresh today."
    "This story was very warm."
    "The pacing worked very well."
    "I loved this movie completely."
    "The dialogue felt sharp throughout."
];

negative_sentences = [
    "The movie was very bad."
    "This film was really awful."
    "I found the story boring."
    "The acting felt quite wooden."
    "This movie looked very cheap."
    "The ending was truly annoying."
    "I disliked this film today."
    "The plot felt weak throughout."
    "This was a dreadful movie."
    "The cast gave poor performances."
    "I hated the final scene."
    "This film felt deeply empty."
    "The soundtrack was very unpleasant."
    "The script was dumb overall."
    "I found it quite painful."
    "The movie felt stale today."
    "This story was very dull."
    "The pacing worked very poorly."
    "I hated this movie completely."
    "The dialogue felt flat throughout."
];

sentences = [positive_sentences; negative_sentences];

%% =========================================================
% 3) PROYECTAR IMPORTANCIAS POR POSICIÓN A TOKENS REALES
% ==========================================================
% [CLS] -> posición 1
% palabra 1 -> posición 2
% palabra 2 -> posición 3
% palabra 3 -> posición 4
% palabra 4 -> posición 5
% palabra 5 -> posición 6
% . -> posición 7
% [SEP] -> posición 8
% ==========================================================

occ_tokens = strings(0,1);
occ_omeda  = [];
occ_plsda  = [];
occ_asca   = [];

for i = 1:numel(sentences)

    toks = simple_tokenize(sentences(i));

    for k = 1:numel(toks)

        bert_pos = k + 1; % palabra 1 va a Token 2

        occ_tokens(end+1,1) = toks(k);
        occ_omeda(end+1,1)  = omeda_pos(bert_pos);
        occ_plsda(end+1,1)  = plsda_pos(bert_pos);
        occ_asca(end+1,1)   = asca_pos(bert_pos);

    end
end

%% =========================================================
% 4) SELECCIONAR LOS TOKENS MÁS IMPORTANTES SEGÚN SHAP
% ==========================================================

% Quitamos tokens especiales o vacíos si aparecen
valid = shap_tokens ~= "" & shap_tokens ~= "." & ...
        shap_tokens ~= "[cls]" & shap_tokens ~= "[sep]";

tokens_valid = shap_tokens(valid);
shap_valid   = shap_values(valid);

% Quedarse solo con tokens que aparecen en las frases utilizadas
tokens_presentes = unique(occ_tokens);

is_present = ismember(tokens_valid, tokens_presentes);

tokens_valid = tokens_valid(is_present);
shap_valid   = shap_valid(is_present);

% Ordenar por importancia SHAP
[~, order] = sort(shap_valid, 'descend');

N = 15;
N = min(N, numel(order));

top_tokens = tokens_valid(order(1:N));
top_shap   = shap_valid(order(1:N));

%% =========================================================
% 5) CALCULAR IMPORTANCIA DE CADA MÉTODO PARA ESOS TOKENS
% ==========================================================

omeda_tok = zeros(N,1);
plsda_tok = zeros(N,1);
asca_tok  = zeros(N,1);

for i = 1:N

    tok = top_tokens(i);

    idx_occ = occ_tokens == tok;

    if any(idx_occ)
        omeda_tok(i) = mean(occ_omeda(idx_occ));
        plsda_tok(i) = mean(occ_plsda(idx_occ));
        asca_tok(i)  = mean(occ_asca(idx_occ));
    else
        omeda_tok(i) = NaN;
        plsda_tok(i) = NaN;
        asca_tok(i)  = NaN;
    end

end

%% =========================================================
% 6) NORMALIZAR PARA COMPARAR PERFILES RELATIVOS
% ==========================================================

M = [omeda_tok, plsda_tok, asca_tok, top_shap];

% Normalización por método
M = 100 * M ./ sum(M, 1, 'omitnan');

%% =========================================================
% 7) FIGURA DE BARRAS HORIZONTALES
% ==========================================================

figure('Color','w','Position',[100 100 1300 800]);

% Invertimos para que el token más importante de SHAP quede arriba
barh(flipud(M), 'grouped');

yticks(1:N);
yticklabels(flipud(top_tokens));

xlabel('Importancia relativa normalizada (%)', ...
       'FontSize', 16, ...
       'FontWeight', 'bold');

ylabel('Token', ...
       'FontSize', 16, ...
       'FontWeight', 'bold');

title('Comparación de tokens más importantes según SHAP', ...
      'FontSize', 20, ...
      'FontWeight', 'bold');

legend({'oMEDA','PLS-DA','ASCA','SHAP'}, ...
       'Location','southeast', ...
       'FontSize', 13);

grid on;
set(gca, 'FontSize', 13, 'LineWidth', 1.1);

exportgraphics(gcf, 'comparacion_tokens_reales_shap_omeda_plsda_asca.png', ...
               'Resolution', 300);

%% =========================================================
% 8) GUARDAR TABLA FINAL
% ==========================================================

tabla = table(top_tokens, M(:,1), M(:,2), M(:,3), M(:,4), ...
    'VariableNames', {'Token','oMEDA','PLS_DA','ASCA','SHAP'});

writetable(tabla, 'comparacion_tokens_reales_shap_omeda_plsda_asca.csv');

disp(tabla);

%% =========================================================
% FUNCIÓN LOCAL
% ==========================================================

function toks = simple_tokenize(sentence)

    sentence = lower(string(sentence));
    sentence = erase(sentence, ",");
    sentence = replace(sentence, ".", " .");
    toks = split(strtrim(sentence));
    toks = toks(toks ~= "");

end