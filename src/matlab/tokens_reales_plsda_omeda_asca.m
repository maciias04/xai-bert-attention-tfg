%% =========================================================
% IMPORTANCIA POR TOKEN REAL
% oMEDA, PLS-DA y ASCA
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

% Usamos la importancia combinada: origen + destino
imp_omeda_pos = T_omeda.Combinada(:);
imp_plsda_pos = T_plsda.Combinada(:);
imp_asca_pos  = T_asca.Combinada(:);

%% =========================================================
% 2) FRASES DEL CONJUNTO EXPERIMENTAL
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
% 3) PROYECTAR IMPORTANCIA POR POSICIÓN A TOKEN REAL
% ==========================================================
% Posiciones:
% Token 1 -> [CLS]
% Token 2 -> palabra 1
% Token 3 -> palabra 2
% Token 4 -> palabra 3
% Token 5 -> palabra 4
% Token 6 -> palabra 5
% Token 7 -> .
% Token 8 -> [SEP]
% ==========================================================

[tokens_omeda, values_omeda] = project_position_importance_to_real_tokens(sentences, imp_omeda_pos);
[tokens_plsda, values_plsda] = project_position_importance_to_real_tokens(sentences, imp_plsda_pos);
[tokens_asca,  values_asca]  = project_position_importance_to_real_tokens(sentences, imp_asca_pos);

%% =========================================================
% 4) GUARDAR TABLAS
% ==========================================================

T_real_omeda = table(tokens_omeda, values_omeda, ...
    'VariableNames', {'Token','Importancia_oMEDA'});

T_real_plsda = table(tokens_plsda, values_plsda, ...
    'VariableNames', {'Token','Importancia_PLS_DA'});

T_real_asca = table(tokens_asca, values_asca, ...
    'VariableNames', {'Token','Importancia_ASCA'});

writetable(T_real_omeda, 'omeda_importancia_tokens_reales.csv');
writetable(T_real_plsda, 'plsda_importancia_tokens_reales.csv');
writetable(T_real_asca,  'asca_importancia_tokens_reales.csv');

%% =========================================================
% 5) GENERAR FIGURAS INDIVIDUALES
% ==========================================================

TopN = 15;

plot_real_token_importance(tokens_omeda, values_omeda, TopN, ...
    'oMEDA: importancia por token real', ...
    'omeda_importancia_tokens_reales.png');

plot_real_token_importance(tokens_plsda, values_plsda, TopN, ...
    'PLS-DA: importancia por token real', ...
    'plsda_importancia_tokens_reales.png');

plot_real_token_importance(tokens_asca, values_asca, TopN, ...
    'ASCA: importancia por token real', ...
    'asca_importancia_tokens_reales.png');

disp('Figuras generadas correctamente.');
disp('Tablas generadas correctamente.');

%% =========================================================
% FUNCIONES LOCALES
% ==========================================================

function [unique_tokens, importance_norm] = project_position_importance_to_real_tokens(sentences, position_importance)

    all_tokens = strings(0,1);
    all_values = [];

    for i = 1:numel(sentences)

        toks = simple_tokenize(sentences(i));

        for k = 1:numel(toks)

            % palabra 1 corresponde a Token 2
            bert_pos = k + 1;

            if bert_pos <= numel(position_importance)
                all_tokens(end+1,1) = toks(k);
                all_values(end+1,1) = position_importance(bert_pos);
            end
        end
    end

    % Eliminar punto si se quiere analizar solo palabras reales
    keep = all_tokens ~= ".";
    all_tokens = all_tokens(keep);
    all_values = all_values(keep);

    unique_tokens = unique(all_tokens);

    importance = zeros(numel(unique_tokens),1);

    for j = 1:numel(unique_tokens)
        idx = all_tokens == unique_tokens(j);
        importance(j) = mean(all_values(idx));
    end

    % Normalizar al 100 %
    importance_norm = 100 * importance / sum(importance);

    % Ordenar de mayor a menor
    [importance_norm, order] = sort(importance_norm, 'descend');
    unique_tokens = unique_tokens(order);

end

function toks = simple_tokenize(sentence)

    sentence = lower(string(sentence));
    sentence = erase(sentence, ",");
    sentence = replace(sentence, ".", " .");

    toks = split(strtrim(sentence));
    toks = toks(toks ~= "");

end

function plot_real_token_importance(tokens, values, TopN, fig_title, filename)

    N = min(TopN, numel(tokens));

    tokens_plot = tokens(1:N);
    values_plot = values(1:N);

    figure('Color','w','Position',[100 100 1100 750]);

    barh(flipud(values_plot));

    yticks(1:N);
    yticklabels(flipud(tokens_plot));

    xlabel('Importancia relativa normalizada (%)', ...
        'FontSize', 16, ...
        'FontWeight', 'bold');

    ylabel('Token real', ...
        'FontSize', 16, ...
        'FontWeight', 'bold');

    title(fig_title, ...
        'FontSize', 20, ...
        'FontWeight', 'bold');

    grid on;
    set(gca, 'FontSize', 13, 'LineWidth', 1.1);

    exportgraphics(gcf, filename, 'Resolution', 300);

end