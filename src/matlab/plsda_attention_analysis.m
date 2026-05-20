%% =========================================================
%  PLS-DA sobre la matriz de atención de BERT
%  Autor: Adrián Macias Caballero
%
%  Objetivo:
%  - Repetir el análisis multivariante usando PLS-DA
%  - Comparar con PCA + oMEDA
%  - Obtener scores, matriz de confusión e importancia por capa/head
% ==========================================================

clear;
clc;
close all;
% ==========================================================
% ESTILO GENERAL DE FIGURAS
% ==========================================================

FS_TITULO  = 22;
FS_EJES    = 18;
FS_TICKS   = 15;
FS_LEYENDA = 16;

FIG_POS = [100 100 1200 700];
AX_POS  = [0.10 0.14 0.84 0.72];
%% =========================================================
% 1) CARGAR DATOS
% ==========================================================

load('pca_omeda_attention_results.mat');

if ~exist('X', 'var') && exist('x', 'var')
    X = x;
end

labels = string(true_label);
labels = labels(:);

X = double(X);

[n_obs, n_vars] = size(X);

fprintf('Número de observaciones: %d\n', n_obs);
fprintf('Número de variables: %d\n', n_vars);

disp('Etiquetas encontradas:');
disp(unique(labels));

%% =========================================================
% 2) CODIFICAR ETIQUETAS PARA PLS-DA
% ==========================================================
% positive -> 1
% negative -> 0
% ==========================================================

Y = double(labels == "positive");
Y = Y(:);

fprintf('Frases positivas: %d\n', sum(Y == 1));
fprintf('Frases negativas: %d\n', sum(Y == 0));
%% =========================================================
% 3) ESTANDARIZAR X
% ==========================================================
% Igual que en PCA, estandarizamos las variables.
% ==========================================================

[Xz, muX, sigmaX] = zscore(X);

% Evitar problemas si alguna variable tiene desviación típica cero
sigmaX(sigmaX == 0) = 1;
Xz = (X - muX) ./ sigmaX;

%% =========================================================
% 4) SELECCIÓN DEL NÚMERO DE COMPONENTES MEDIANTE LOOCV
% ==========================================================
% Como solo tenemos 40 frases, usamos Leave-One-Out Cross Validation.
% Esto evita elegir componentes solo mirando el ajuste en entrenamiento.
% ==========================================================

maxComp = 10;

accuracy_cv = zeros(maxComp, 1);
rmse_cv = zeros(maxComp, 1);

for ncomp = 1:maxComp

    Y_pred_cv = zeros(n_obs, 1);

    for i = 1:n_obs

        % Separar train/test
        idx_test = i;
        idx_train = setdiff(1:n_obs, idx_test);

        X_train = Xz(idx_train, :);
        Y_train = Y(idx_train);

        X_test = Xz(idx_test, :);

        % Ajustar PLS con ncomp componentes
        [~, ~, ~, ~, beta_cv] = plsregress(X_train, Y_train, ncomp);

        % Predecir muestra dejada fuera
        Y_pred_cv(i) = [1 X_test] * beta_cv;
    end

    % Clasificación con umbral 0.5
    Y_class_cv = Y_pred_cv >= 0.5;

    accuracy_cv(ncomp) = mean(Y_class_cv == Y);
    rmse_cv(ncomp) = sqrt(mean((Y - Y_pred_cv).^2));

end

% Mostrar resultados
disp(table((1:maxComp)', accuracy_cv, rmse_cv, ...
    'VariableNames', {'Componentes', 'Accuracy_CV', 'RMSE_CV'}));

% Elegir número óptimo:
% criterio simple -> máxima accuracy y, en empate, menor número de componentes
bestAcc = max(accuracy_cv);
bestComp = find(accuracy_cv == bestAcc, 1, 'first');

fprintf('\nNúmero óptimo de componentes seleccionado: %d\n', bestComp);
fprintf('Accuracy LOOCV: %.2f %%\n', 100 * bestAcc);

%% =========================================================
% 5) GRÁFICA DE VALIDACIÓN CRUZADA
% ==========================================================

figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

plot(ax, 1:maxComp, accuracy_cv, '-o', ...
     'LineWidth', 1.8, ...
     'MarkerSize', 7);

xlabel(ax, 'Número de componentes PLS', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Accuracy LOOCV', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('Selección del número de componentes en PLS-DA', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');

saveas(gcf, 'plsda_accuracy_cv.png');

figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

plot(ax, 1:maxComp, rmse_cv, '-o', ...
     'LineWidth', 1.8, ...
     'MarkerSize', 7);

xlabel(ax, 'Número de componentes PLS', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'RMSE LOOCV', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('Error de validación cruzada en PLS-DA', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');

saveas(gcf, 'plsda_rmse_cv.png');

%% =========================================================
% 6) AJUSTAR MODELO FINAL PLS-DA
% ==========================================================

[XL, YL, XS, YS, beta, PCTVAR, MSE, stats] = plsregress(Xz, Y, bestComp);

% Predicción sobre entrenamiento
Y_pred = [ones(n_obs, 1) Xz] * beta;
Y_class = Y_pred >= 0.5;

accuracy_train = mean(Y_class == Y);

fprintf('\nAccuracy en entrenamiento: %.2f %%\n', 100 * accuracy_train);

%% =========================================================
% 7) SCORE PLOT PLS-DA
% ==========================================================

figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

gscatter(ax, XS(:,1), XS(:,2), labels, 'br', 'ox', 8);

xlabel(ax, 'LV1', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'LV2', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('PLS-DA: representación de las observaciones en LV1-LV2', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

leyenda = legend(ax, 'Location', 'best');
leyenda.FontSize = FS_LEYENDA;
leyenda.FontWeight = 'bold';

grid(ax, 'on');

saveas(gcf, 'plsda_scores_lv1_lv2.png');

%% =========================================================
% 8) MATRIZ DE CONFUSIÓN
% ==========================================================

figure('Color','w','Position',FIG_POS);

cm = confusionchart(categorical(Y), categorical(Y_class), ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

cm.FontSize = FS_TICKS;
cm.Title = '';

sgtitle('Matriz de confusión PLS-DA', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

saveas(gcf, 'plsda_confusion_matrix.png');

%% =========================================================
% 9) PREDICCIÓN CONTINUA DEL MODELO
% ==========================================================

figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

hold(ax, 'on');

idx_pos = find(Y == 1);
idx_neg = find(Y == 0);

scatter(ax, idx_pos, Y_pred(idx_pos), 70, 'b', 'filled');
scatter(ax, idx_neg, Y_pred(idx_neg), 70, 'r', 'filled');

hthr = yline(ax, 0.5, '--k', 'Umbral 0.5');
hthr.FontSize = FS_LEYENDA;
hthr.FontWeight = 'bold';
hthr.LabelHorizontalAlignment = 'left';
hthr.LabelVerticalAlignment = 'bottom';

xlabel(ax, 'Índice de frase', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Salida continua PLS-DA', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('Predicción continua del modelo PLS-DA', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

leyenda = legend(ax, 'Positivas', 'Negativas', 'Umbral', ...
                 'Location', 'best');
leyenda.FontSize = FS_LEYENDA;
leyenda.FontWeight = 'bold';

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');
hold(ax, 'off');

saveas(gcf, 'plsda_prediccion_continua.png');

%% =========================================================
% 10) IMPORTANCIA DE VARIABLES MEDIANTE COEFICIENTES
% ==========================================================
% beta(1) es el intercepto.
% beta(2:end) corresponde a las 9216 variables originales.
% ==========================================================

coef_pls = beta(2:end);
importance_coef = abs(coef_pls);

% Ranking de variables
[importance_coef_sorted, idx_sorted_coef] = sort(importance_coef, 'descend');

K = 50;
topK_idx_coef = idx_sorted_coef(1:K);
topK_vals_coef = coef_pls(topK_idx_coef);

fprintf('\nTop %d variables según coeficientes PLS-DA:\n', K);
disp(table(topK_idx_coef, topK_vals_coef));

writetable(table(topK_idx_coef, topK_vals_coef), ...
    'plsda_top50_variables_coeficientes.csv');

%% =========================================================
% 11) HISTOGRAMA DE IMPORTANCIA POR COEFICIENTES
% ==========================================================

figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

histogram(ax, importance_coef, 50);

xlabel(ax, '|Coeficiente PLS-DA|', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Frecuencia', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('Distribución de importancia de variables en PLS-DA', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');

saveas(gcf, 'plsda_hist_importancia_coeficientes.png');

%% =========================================================
% 12) VIP SCORES
% ==========================================================

vip_scores = calculate_vip(Xz, Y, XS, YL, stats.W);

[VIP_sorted, idx_sorted_vip] = sort(vip_scores, 'descend');

topK_idx_vip = idx_sorted_vip(1:K);
topK_vals_vip = VIP_sorted(1:K);

fprintf('\nTop %d variables según VIP:\n', K);
disp(table(topK_idx_vip, topK_vals_vip));

writetable(table(topK_idx_vip, topK_vals_vip), ...
    'plsda_top50_variables_vip.csv');

figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

histogram(ax, vip_scores, 50);

xlabel(ax, 'VIP score', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Frecuencia', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('Distribución de VIP scores', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');
hold(ax, 'on');

hvip = xline(ax, 1, '--r', 'VIP = 1');
hvip.FontSize = FS_LEYENDA;
hvip.FontWeight = 'bold';
hvip.LabelVerticalAlignment = 'top';
hvip.LabelHorizontalAlignment = 'left';

hold(ax, 'off');

saveas(gcf, 'plsda_hist_vip.png');

%% =========================================================
% 13) IMPORTANCIA POR CAPA USANDO VIP
% ==========================================================
% 9216 = 12 capas * 12 heads * 8 * 8
% ==========================================================

imp_by_layer_pls = zeros(12, 1);
imp_by_head_pls = zeros(12, 12);

for var_idx = 1:n_vars

    zero_idx = var_idx - 1;

    layer = floor(zero_idx / (12 * 64));
    rem1 = mod(zero_idx, 12 * 64);

    head = floor(rem1 / 64);

    imp_by_layer_pls(layer + 1) = imp_by_layer_pls(layer + 1) + vip_scores(var_idx);
    imp_by_head_pls(layer + 1, head + 1) = imp_by_head_pls(layer + 1, head + 1) + vip_scores(var_idx);
end

% Gráfica por capa
figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

bar(ax, 0:11, imp_by_layer_pls);

xlabel(ax, 'Layer', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Importancia acumulada VIP', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('PLS-DA: importancia por capa', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

xticks(ax, 0:11);
grid(ax, 'on');

saveas(gcf, 'plsda_importancia_por_capa.png');

% Heatmap layer-head
figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = [0.10 0.14 0.76 0.72];

imagesc(ax, 0:11, 0:11, imp_by_head_pls);

cb = colorbar(ax);
cb.FontSize = FS_TICKS;
cb.Label.String = 'Importancia acumulada VIP';
cb.Label.FontSize = FS_EJES;
cb.Label.FontWeight = 'bold';

xlabel(ax, 'Head', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Layer', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('PLS-DA: importancia por layer-head', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'YDir', 'normal', ...
        'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

xticks(ax, 0:11);
yticks(ax, 0:11);

saveas(gcf, 'plsda_importancia_layer_head.png');

%% =========================================================
% 14) COMPARACIÓN CON oMEDA, SI EXISTE omeda_vec
% ==========================================================
% Si tienes omeda_vec cargado en el .mat o generado antes, este bloque
% compara la importancia por capas entre oMEDA y PLS-DA.
% ==========================================================

if exist('omeda_vec', 'var')

    importance_omeda = abs(omeda_vec);

    imp_by_layer_omeda = zeros(12, 1);

    for var_idx = 1:n_vars

        zero_idx = var_idx - 1;

        layer = floor(zero_idx / (12 * 64));

        imp_by_layer_omeda(layer + 1) = imp_by_layer_omeda(layer + 1) + importance_omeda(var_idx);
    end

    % Normalizar para comparar formas
    imp_by_layer_omeda_norm = imp_by_layer_omeda / sum(imp_by_layer_omeda);
    imp_by_layer_pls_norm = imp_by_layer_pls / sum(imp_by_layer_pls);

    figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

bar(ax, 0:11, [imp_by_layer_omeda_norm, imp_by_layer_pls_norm]);

xlabel(ax, 'Layer', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Importancia normalizada', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('Comparación por capas: oMEDA vs PLS-DA', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

leyenda = legend(ax, 'oMEDA', 'PLS-DA', ...
                 'Location', 'best');
leyenda.FontSize = FS_LEYENDA;
leyenda.FontWeight = 'bold';

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

xticks(ax, 0:11);
grid(ax, 'on');

saveas(gcf, 'comparacion_omeda_plsda_por_capa.png');

    % Correlación variable a variable
    corr_var = corr(importance_omeda(:), vip_scores(:), 'Type', 'Spearman');

    fprintf('\nCorrelación Spearman entre |oMEDA| y VIP PLS-DA: %.4f\n', corr_var);

    % Comparación Top-K
    topK_omeda = find_topK(importance_omeda, K);
    topK_pls = find_topK(vip_scores, K);

    overlap_topK = numel(intersect(topK_omeda, topK_pls));

    fprintf('Solapamiento Top-%d entre oMEDA y PLS-DA: %d variables\n', K, overlap_topK);

    figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

scatter(ax, importance_omeda, vip_scores, 20, 'filled');

xlabel(ax, '|oMEDA|', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'VIP PLS-DA', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('Comparación variable a variable: oMEDA vs PLS-DA', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');

saveas(gcf, 'scatter_omeda_vs_plsda.png');

else
    fprintf('\nNo se encontró omeda_vec. Se omite comparación directa con oMEDA.\n');
end

%% =========================================================
% 15) GUARDAR RESULTADOS PRINCIPALES
% ==========================================================

save('plsda_results.mat', ...
    'bestComp', ...
    'accuracy_cv', ...
    'rmse_cv', ...
    'accuracy_train', ...
    'XL', 'YL', 'XS', 'YS', ...
    'beta', 'PCTVAR', 'MSE', 'stats', ...
    'Y', 'Y_pred', 'Y_class', ...
    'coef_pls', 'importance_coef', ...
    'vip_scores', ...
    'imp_by_layer_pls', ...
    'imp_by_head_pls');

fprintf('\nAnálisis PLS-DA finalizado.\n');
fprintf('Resultados guardados en plsda_results.mat\n');

%% =========================================================
% FUNCIONES AUXILIARES
% ==========================================================

function vip = calculate_vip(X, Y, T, Q, W)
    % Calcula VIP scores para un modelo PLS con una salida Y.
    %
    % X -> matriz de datos estandarizada
    % Y -> vector respuesta
    % T -> scores X
    % Q -> loadings Y
    % W -> weights X

    [~, p] = size(X);
    A = size(T, 2);

    % Asegurar que Y está centrada
    Yc = Y - mean(Y);

    % Suma de cuadrados explicada por cada componente
    SSY = zeros(A, 1);

    for a = 1:A
        ta = T(:, a);
        qa = Q(a);
        SSY(a) = sum((ta * qa).^2);
    end

    total_SSY = sum(SSY);

    vip = zeros(p, 1);

    for j = 1:p
        weight_sum = 0;

        for a = 1:A
            wa = W(:, a);
            weight_sum = weight_sum + SSY(a) * (W(j, a)^2 / sum(wa.^2));
        end

        vip(j) = sqrt(p * weight_sum / total_SSY);
    end
end

function idx_top = find_topK(values, K)
    [~, idx_sorted] = sort(values, 'descend');
    idx_top = idx_sorted(1:K);
end

%% =========================================================
% 16) IMPORTANCIA POR TOKEN EN PLS-DA
% ==========================================================
% Se utiliza VIP score como medida de importancia de cada variable.
% Después se acumula por token origen, token destino y combinación.
% ==========================================================

n_tokens = 8;
n_heads  = 12;

token_labels = ["[CLS]"; "Token 2"; "Token 3"; "Token 4"; ...
                "Token 5"; "Token 6"; "."; "[SEP]"];

importance_plsda = vip_scores(:);

imp_tokenO_plsda = zeros(n_tokens,1);
imp_tokenD_plsda = zeros(n_tokens,1);

for var_idx = 1:n_vars

    zero_idx = var_idx - 1;

    layer_idx = floor(zero_idx / (n_heads*n_tokens*n_tokens));
    rem1      = mod(zero_idx, n_heads*n_tokens*n_tokens);

    head_idx  = floor(rem1 / (n_tokens*n_tokens));
    rem2      = mod(rem1, n_tokens*n_tokens);

    tokenO_idx = floor(rem2 / n_tokens) + 1;
    tokenD_idx = mod(rem2, n_tokens) + 1;

    imp_tokenO_plsda(tokenO_idx) = imp_tokenO_plsda(tokenO_idx) + importance_plsda(var_idx);
    imp_tokenD_plsda(tokenD_idx) = imp_tokenD_plsda(tokenD_idx) + importance_plsda(var_idx);
end

imp_token_comb_plsda = imp_tokenO_plsda + imp_tokenD_plsda;

imp_tokenO_plsda_pct     = 100 * imp_tokenO_plsda     / sum(imp_tokenO_plsda);
imp_tokenD_plsda_pct     = 100 * imp_tokenD_plsda     / sum(imp_tokenD_plsda);
imp_token_comb_plsda_pct = 100 * imp_token_comb_plsda / sum(imp_token_comb_plsda);

tabla_plsda_tokens = table(token_labels, ...
    imp_tokenO_plsda_pct, ...
    imp_tokenD_plsda_pct, ...
    imp_token_comb_plsda_pct, ...
    'VariableNames', {'Token','Origen','Destino','Combinada'});

writetable(tabla_plsda_tokens, 'plsda_importancia_tokens.csv');

figure('Color','w','Position',[100 100 1200 700]);

bar([imp_tokenO_plsda_pct, imp_tokenD_plsda_pct, imp_token_comb_plsda_pct]);

xticks(1:n_tokens);
xticklabels(token_labels);

xlabel('Token', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Importancia relativa (%)', 'FontSize', 18, 'FontWeight', 'bold');

legend({'Como origen','Como destino','Combinada'}, ...
       'Location','best', ...
       'FontSize', 14);

title('PLS-DA: importancia relativa por token', ...
      'FontSize', 22, ...
      'FontWeight', 'bold');

grid on;
set(gca, 'FontSize', 15, 'LineWidth', 1.2);

exportgraphics(gcf, 'plsda_importancia_tokens.png', 'Resolution', 300);