%% =========================================================
% COMPARACIÓN FINAL DE IMPORTANCIA POR TOKEN
% oMEDA vs PLS-DA vs ASCA vs SHAP
% ==========================================================

clear;
close all;
clc;

%% =========================================================
% 1) CARGAR TABLAS
% ==========================================================

T_omeda = readtable('omeda_importancia_tokens.csv');
T_plsda = readtable('plsda_importancia_tokens.csv');
T_asca  = readtable('asca_importancia_tokens.csv');
T_shap  = readtable('shap_importancia_tokens.csv');

%% =========================================================
% 2) EXTRAER IMPORTANCIAS
% ==========================================================
% Para oMEDA, PLS-DA y ASCA usamos la columna "Combinada"
% Para SHAP usamos la columna "Importancia"
% ==========================================================

omeda_imp = T_omeda.Combinada(:);
plsda_imp = T_plsda.Combinada(:);
asca_imp  = T_asca.Combinada(:);
shap_imp  = T_shap.Importancia(:);

%% =========================================================
% 3) ELEGIR TOKENS A COMPARAR
% ==========================================================
% Opción recomendada:
% comparar solo tokens 2:7
% porque [CLS] y [SEP] no son realmente comparables con SHAP
% ==========================================================

idx = 1:8;

token_labels = string(T_omeda.Token(idx));

M = [omeda_imp(idx), plsda_imp(idx), asca_imp(idx), shap_imp(idx)];

%% =========================================================
% 4) NORMALIZAR CADA MÉTODO AL 100%
% ==========================================================
% Así comparamos perfiles relativos, no magnitudes absolutas
% ==========================================================

M = 100 * M ./ sum(M,1);

%% =========================================================
% 5) FIGURA PRINCIPAL: BARRAS AGRUPADAS
% ==========================================================

figure('Color','w','Position',[100 100 1400 800]);

bar(M, 'grouped');

xticks(1:length(token_labels));
xticklabels(token_labels);

xlabel('Posición de token', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Importancia relativa normalizada (%)', 'FontSize', 18, 'FontWeight', 'bold');

title('Comparación de la importancia relativa por token', ...
      'FontSize', 22, 'FontWeight', 'bold');

legend({'oMEDA','PLS-DA','ASCA','SHAP'}, ...
       'Location','best', ...
       'FontSize', 15);

grid on;
set(gca, 'FontSize', 15, 'LineWidth', 1.2);

exportgraphics(gcf, 'comparacion_importancia_tokens_metodos.png', 'Resolution', 300);

%% =========================================================
% 6) GUARDAR TABLA FINAL
% ==========================================================

tabla_comparacion = array2table(M, ...
    'VariableNames', {'oMEDA','PLS_DA','ASCA','SHAP'});

tabla_comparacion.Token = token_labels;
tabla_comparacion = movevars(tabla_comparacion, 'Token', 'Before', 1);

writetable(tabla_comparacion, 'comparacion_importancia_tokens_metodos.csv');

disp('Figura guardada: comparacion_importancia_tokens_metodos.png');
disp('Tabla guardada: comparacion_importancia_tokens_metodos.csv');
disp(tabla_comparacion);