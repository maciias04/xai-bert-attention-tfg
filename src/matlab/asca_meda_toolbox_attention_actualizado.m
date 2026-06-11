%% =========================================================
%  ASCA sobre la matriz de atención de BERT usando MEDA Toolbox
%  Autor: Adrián Macias Caballero
%
%  Objetivo:
%  - Usar ASCA con las funciones de la MEDA Toolbox de José Camacho
%  - Evitar implementaciones manuales del efecto positivo-negativo
%  - Partir del flujo nuevo usado en PCA/oMEDA/PLS-DA
%  - Analizar la importancia por variable, capa, head y token
% ==========================================================

clear;
clc;
close all;

%% =========================================================
% 0) AÑADIR MEDA TOOLBOX AL PATH
% ==========================================================
% Cambia esta ruta por la ruta real de tu toolbox si hace falta.
% ==========================================================

addpath(genpath('C:\Users\Usuario\Desktop\UNI\Cuarto de Carrera Teleco\doctoradopepe\MEDA Shared Materials-20260301T114916Z-3-001\MEDA Shared Materials\MEDA-Toolbox-master\MEDA-Toolbox-master\toolbox'));
savepath;

disp('Comprobando funciones de MEDA Toolbox...');

if isempty(which('parglm'))
    error('No se encuentra parglm. Añade la MEDA Toolbox al path de MATLAB.');
end

if isempty(which('asca'))
    error('No se encuentra asca. Añade la MEDA Toolbox al path de MATLAB.');
end

if isempty(which('scores'))
    error('No se encuentra scores. Añade la MEDA Toolbox al path de MATLAB.');
end

if isempty(which('loadings'))
    error('No se encuentra loadings. Añade la MEDA Toolbox al path de MATLAB.');
end

disp('MEDA Toolbox encontrada correctamente.');

%% =========================================================
% 1) ESTILO GENERAL DE FIGURAS
% ==========================================================

FS_TITULO  = 22;
FS_EJES    = 18;
FS_TICKS   = 15;
FS_LEYENDA = 16;

FIG_POS = [100 100 1200 700];
AX_POS  = [0.10 0.14 0.84 0.72];

rng(42);

%% =========================================================
% 2) CARGAR DATOS
% ==========================================================
% Se prioriza el archivo generado por el PCA con MEDA Toolbox para mantener
% coherencia con el resto del flujo. Si no existe, se carga el CSV original.
% ==========================================================

if exist('pca_attention_results_meda.mat', 'file')

    load('pca_attention_results_meda.mat');

    if ~exist('X', 'var')
        error('El archivo pca_attention_results_meda.mat no contiene la matriz X.');
    end

    X = double(X);

    if exist('true_label', 'var')
        labels = lower(string(true_label(:)));
    else
        labels = [repmat("positive",20,1); repmat("negative",20,1)];
    end

else

    data = importdata('attention_matrix_40x9216.csv');
    X = double(data.data);
    labels = [repmat("positive",20,1); repmat("negative",20,1)];

end

[n_obs, n_vars] = size(X);

fprintf('Número de observaciones: %d\n', n_obs);
fprintf('Número de variables: %d\n', n_vars);

%% =========================================================
% 3) DEFINIR FACTORES DEL DISEÑO EXPERIMENTAL
% ==========================================================
% ASCA necesita una matriz de diseño F.
%
% Factor 1: clase de sentimiento
%   1 -> positive
%   2 -> negative
%
% Factor 2: pareja de frase
%   1..20 -> pareja positivo-negativo equivalente
%
% El segundo factor permite considerar que las frases están organizadas en
% parejas semánticas, como en el diseño original.
% ==========================================================

idx_pos = labels == "positive" | labels == "positiva" | labels == "positivas";
idx_neg = labels == "negative" | labels == "negativa" | labels == "negativas";

if sum(idx_pos) ~= sum(idx_neg)
    warning('El número de frases positivas y negativas no coincide. Se construirá el factor pareja en función del orden disponible.');
end

if ~any(idx_pos) || ~any(idx_neg)
    error('No se han encontrado correctamente las clases positive/negative.');
end
% (factor 1)
class_id = zeros(n_obs,1);%matrix 40 x 1 de zeros que se rellena de 1 para las positivas y de 2 para las negativas 
class_id(idx_pos) = 1;
class_id(idx_neg) = 2;

n_pos = sum(idx_pos);%calcula el numero de frases positivas
n_neg = sum(idx_neg);%calcula el numero de frases negativas
n_pair = min(n_pos, n_neg);%calcula el numero de frases pareadas

pair_id = zeros(n_obs,1);%matriz 40 x 1 para asignarle un orden a las pareadas(es el factor 2)
idx_pos_order = find(idx_pos);
idx_neg_order = find(idx_neg);

pair_id(idx_pos_order(1:n_pair)) = (1:n_pair)';
pair_id(idx_neg_order(1:n_pair)) = (1:n_pair)';

% Si hubiera observaciones extra no emparejadas, se les asigna un índice nuevo.
idx_unpaired = find(pair_id == 0);
if ~isempty(idx_unpaired)
    pair_id(idx_unpaired) = n_pair + (1:numel(idx_unpaired))';
end

F = [class_id, pair_id];%matriz F con los dos factores 40 x 2

fprintf('Frases positivas: %d\n', sum(class_id == 1));
fprintf('Frases negativas: %d\n', sum(class_id == 2));
fprintf('Número de parejas/bloques: %d\n', numel(unique(pair_id)));

%% =========================================================
% 4) EJECUTAR PARGLM + ASCA CON MEDA TOOLBOX
% ==========================================================
% parglm realiza la descomposición del diseño experimental.
% asca aplica PCA a las matrices de efecto obtenidas por parglm.
%
% Preprocessing = 2 -> autoescalado, igual que en PCA/oMEDA/PLS-DA.
% Model = 'linear' -> se consideran los efectos principales clase y pareja.
% ==========================================================

[tabla_anova_asca, parglmo] = parglm(X, F, ...%Esta función toma tus datos experimentales y el diseño de factores, y calcula una descomposición tipo GLM/ANOVA multivariante.
    'Model', 'linear', ...%modelo lineal de efectos principales.
    'Preprocessing', 2, ...%autoescalado
    'Permutations', 1000, ...
    'Ts', 2);
%parglm intenta separar la matriz X en partes:
% X = efecto clase + efecto pareja + residuo
% La toolbox calcula el efecto real de la clase positiva/negativa.
% Luego mezcla o permuta las etiquetas muchas veces.
% Vuelve a calcular el efecto con etiquetas aleatorias.
% Compara el efecto real con los efectos obtenidos al azar.
% Así puede estimar si el efecto observado es mayor de lo que cabría esperar por casualidad.
disp('Tabla ANOVA/ASCA generada por parglm:');
disp(tabla_anova_asca);%cuánto explica cada factor, qué parte queda como residuo, y los resultados asociados a la evaluación estadística mediante permutaciones.
    %   Source          SumSq       PercSumSq    df    MeanSq      F        Pvalue 
    % _____________    __________    _________    __    ______    ______    ________
    % 
    % {'Factor 1' }         20089     5.5893       1     20089    7.7491    0.000999
    % {'Factor 2' }    2.9008e+05     80.706      19     15267    5.8891    0.000999
    % {'Residuals'}         49257     13.704      19    2592.5       NaN         NaN
    % {'Total'    }    3.5942e+05        100      39      9216       NaN         NaN
ascao = asca(parglmo);%aplica el análisis ASCA sobre esa descomposición.

disp('ASCA calculado correctamente con la MEDA Toolbox.');

%% =========================================================
% 5) VISUALIZAR FACTOR CLASE CON FUNCIONES DE LA TOOLBOX
% ==========================================================
% Factor 1 = clase positivo/negativo.
% Factor 2 = pareja/bloque.
% ==========================================================

factor_clase = ascao.factors{1};%corresponde al primer factor, que en tu diseño es la clase.

n_lvs_clase = size(factor_clase.loads, 2);%Los loadings indican cómo contribuyen las variables originales a las variables latentes del factor.
factor_clase.lvs = 1:min(2, n_lvs_clase);%solo tenemos un lvs porque factor_clase.loads es 9216 x 1

fig_scores = scores(factor_clase, ...
    'Title', 'ASCA MEDA: factor clase', ...
    'ObsClass', class_id, ...
    'ObsLabel', (1:n_obs)', ...
    'BlurIndex', Inf);

if ~isempty(fig_scores)
    exportgraphics(fig_scores(1), 'asca_meda_scores_factor_clase.png', 'Resolution', 300);
end

fig_loadings = loadings(factor_clase, ...
    'Title', 'ASCA MEDA: loadings del factor clase', ...
    'BlurIndex', Inf);

if ~isempty(fig_loadings)
    exportgraphics(fig_loadings(1), 'asca_meda_loadings_factor_clase.png', 'Resolution', 300);
end

%% =========================================================
% 6) IMPORTANCIA ASCA POR VARIABLE ORIGINAL
% ==========================================================
% La matriz de efecto del factor clase procede directamente de parglm/asca.
% No se calcula la diferencia media sobre X de forma manual.
%
% Se usa la suma de cuadrados de la matriz de efecto como contribución
% de cada variable al factor clase:
%
%   importancia_j = sum_i X_factor_clase(i,j)^2
%
% El signo se conserva mediante la diferencia entre los niveles del propio
% efecto ASCA: positive - negative.
% ==========================================================

X_factor_clase = ascao.factors{1}.matrix;

effect_signed = mean(X_factor_clase(class_id == 1, :), 1) - ...
                mean(X_factor_clase(class_id == 2, :), 1);

importance_asca = sum(X_factor_clase.^2, 1)';
effect_asca = effect_signed(:);

[importance_sorted, idx_sorted_all] = sort(importance_asca, 'descend');

K = 50;
topK_idx = idx_sorted_all(1:K);
topK_importance = importance_asca(topK_idx);
topK_effect = effect_asca(topK_idx);

tabla_topK = table(topK_idx, topK_effect, topK_importance, ...
    'VariableNames', {'Variable','Efecto_ASCA_pos_menos_neg','Importancia_ASCA'});

disp('Top 50 variables según importancia ASCA del factor clase:');
disp(tabla_topK);

writetable(tabla_topK, 'asca_meda_top50_variables.csv');

%% =========================================================
% 7) DECODIFICAR LAS 9216 VARIABLES DE ATENCIÓN
% ==========================================================
% 9216 = 12 layers * 12 heads * 8 tokens origen * 8 tokens destino.
% Se asume el mismo orden de vectorización usado en Python:
%   layer -> head -> token origen -> token destino
% ==========================================================

n_layers = 12;
n_heads  = 12;
n_tokens = 8;

if n_vars ~= n_layers*n_heads*n_tokens*n_tokens
    error('El número de variables no coincide con 12*12*8*8 = 9216.');
end

token_labels = ["[CLS]"; "Token 2"; "Token 3"; "Token 4"; ...
                "Token 5"; "Token 6"; "."; "[SEP]"];

var_index = (1:n_vars)';
zero_idx = var_index - 1;

layer_idx = floor(zero_idx / (n_heads*n_tokens*n_tokens)) + 1;
rem1      = mod(zero_idx, n_heads*n_tokens*n_tokens);

head_idx  = floor(rem1 / (n_tokens*n_tokens)) + 1;
rem2      = mod(rem1, n_tokens*n_tokens);

tokenO_idx = floor(rem2 / n_tokens) + 1;
tokenD_idx = mod(rem2, n_tokens) + 1;

tabla_variables = table(var_index, layer_idx, head_idx, tokenO_idx, tokenD_idx, ...
    token_labels(tokenO_idx), token_labels(tokenD_idx), ...
    effect_asca, importance_asca, ...
    'VariableNames', {'Variable','Layer','Head','TokenOrigenIdx','TokenDestinoIdx', ...
    'TokenOrigen','TokenDestino','Efecto_ASCA','Importancia_ASCA'});

writetable(tabla_variables, 'asca_meda_variables_interpretadas.csv');

%% =========================================================
% 8) HISTOGRAMA DE IMPORTANCIA ASCA
% ==========================================================

figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

histogram(ax, importance_asca, 50);

xlabel(ax, 'Importancia ASCA', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Frecuencia', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('ASCA MEDA: distribución de importancia de variables', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');

exportgraphics(gcf, 'asca_meda_distribucion_importancia.png', 'Resolution', 300);

%% =========================================================
% 9) IMPORTANCIA POR CAPA
% ==========================================================

imp_by_layer_asca = accumarray(layer_idx, importance_asca, [n_layers 1], @sum, 0);
eff_by_layer_asca = accumarray(layer_idx, effect_asca,      [n_layers 1], @sum, 0);

imp_by_layer_asca_pct = 100 * imp_by_layer_asca / sum(imp_by_layer_asca);

tabla_layer_asca = table((1:n_layers)', imp_by_layer_asca, imp_by_layer_asca_pct, eff_by_layer_asca, ...
    'VariableNames', {'Layer','Importancia_ASCA','Importancia_pct','Efecto_ASCA'});

writetable(tabla_layer_asca, 'asca_meda_importancia_layer.csv');

figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

bar(ax, 1:n_layers, imp_by_layer_asca_pct);

xlabel(ax, 'Layer', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Importancia relativa (%)', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('ASCA MEDA: importancia relativa por capa', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

xticks(ax, 1:n_layers);
grid(ax, 'on');

exportgraphics(gcf, 'asca_meda_importancia_por_capa.png', 'Resolution', 300);

%% =========================================================
% 10) IMPORTANCIA POR HEAD GLOBAL
% ==========================================================

imp_by_head_global_asca = accumarray(head_idx, importance_asca, [n_heads 1], @sum, 0);
eff_by_head_global_asca = accumarray(head_idx, effect_asca,      [n_heads 1], @sum, 0);

imp_by_head_global_asca_pct = 100 * imp_by_head_global_asca / sum(imp_by_head_global_asca);

tabla_head_asca = table((1:n_heads)', imp_by_head_global_asca, imp_by_head_global_asca_pct, eff_by_head_global_asca, ...
    'VariableNames', {'Head','Importancia_ASCA','Importancia_pct','Efecto_ASCA'});

writetable(tabla_head_asca, 'asca_meda_importancia_head.csv');

figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

bar(ax, 1:n_heads, imp_by_head_global_asca_pct);

xlabel(ax, 'Head', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Importancia relativa (%)', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('ASCA MEDA: importancia relativa por head', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

xticks(ax, 1:n_heads);
grid(ax, 'on');

exportgraphics(gcf, 'asca_meda_importancia_por_head.png', 'Resolution', 300);


% Figura adicional en formato clásico: importancia por cabeza de atención.
% Esta figura usa la importancia del factor clase obtenida mediante ASCA
% con parglm/asca. No recalcula ASCA ni usa funciones auxiliares externas.
figure('Color','w','Position',[100 100 900 500]);

ax = axes;
ax.Position = [0.12 0.16 0.82 0.70];

bar(ax, 1:n_heads, imp_by_head_global_asca_pct);

xlabel(ax, 'Attention', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Importancia relativa (%)', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

title(ax, 'ASCA: importancia por attention', ...
      'FontSize', FS_TITULO, ...
      'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

xticks(ax, 1:n_heads);
xticklabels(ax, arrayfun(@(k) sprintf('Attention %d', k), 1:n_heads, 'UniformOutput', false));
xtickangle(ax, 35);

grid(ax, 'on');

exportgraphics(gcf, 'asca_meda_importancia_attention.png', 'Resolution', 300);

%% =========================================================
% 11) HEATMAP LAYER-HEAD
% ==========================================================

imp_by_layer_head_asca = accumarray([layer_idx head_idx], importance_asca, [n_layers n_heads], @sum, 0);

tabla_layer_head_asca = table;
for l = 1:n_layers
    for h = 1:n_heads
        tabla_layer_head_asca = [tabla_layer_head_asca; ...
            table(l, h, imp_by_layer_head_asca(l,h), ...
            'VariableNames', {'Layer','Head','Importancia_ASCA'})];
    end
end

writetable(tabla_layer_head_asca, 'asca_meda_importancia_layer_head.csv');

figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = [0.10 0.14 0.76 0.72];

imagesc(ax, 1:n_heads, 1:n_layers, imp_by_layer_head_asca);

cb = colorbar(ax);
cb.FontSize = FS_TICKS;
cb.Label.String = 'Importancia acumulada ASCA';
cb.Label.FontSize = FS_EJES;
cb.Label.FontWeight = 'bold';

xlabel(ax, 'Head', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Layer', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('ASCA MEDA: importancia por layer-head', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'YDir', 'reverse', ...
        'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

xticks(ax, 1:n_heads);
yticks(ax, 1:n_layers);

xlim(ax, [0.5 12.5]);
ylim(ax, [0.5 12.5]);

grid(ax, 'off');

exportgraphics(gcf, 'asca_meda_importancia_layer_head.png', 'Resolution', 300);

%% =========================================================
% 12) IMPORTANCIA POR TOKEN ORIGEN, DESTINO Y COMBINADA
% ==========================================================

imp_tokenO_asca = accumarray(tokenO_idx, importance_asca, [n_tokens 1], @sum, 0);
imp_tokenD_asca = accumarray(tokenD_idx, importance_asca, [n_tokens 1], @sum, 0);

eff_tokenO_asca = accumarray(tokenO_idx, effect_asca, [n_tokens 1], @sum, 0);
eff_tokenD_asca = accumarray(tokenD_idx, effect_asca, [n_tokens 1], @sum, 0);

imp_token_comb_asca = imp_tokenO_asca + imp_tokenD_asca;

imp_tokenO_asca_pct     = 100 * imp_tokenO_asca     / sum(imp_tokenO_asca);
imp_tokenD_asca_pct     = 100 * imp_tokenD_asca     / sum(imp_tokenD_asca);
imp_token_comb_asca_pct = 100 * imp_token_comb_asca / sum(imp_token_comb_asca);

tabla_tokens_asca = table(token_labels, ...
    imp_tokenO_asca_pct, ...
    imp_tokenD_asca_pct, ...
    imp_token_comb_asca_pct, ...
    eff_tokenO_asca, ...
    eff_tokenD_asca, ...
    'VariableNames', {'Token','Origen_pct','Destino_pct','Combinada_pct','Efecto_origen','Efecto_destino'});

writetable(tabla_tokens_asca, 'asca_meda_importancia_tokens.csv');


% Etiquetas numéricas para reproducir las figuras separadas de token origen
% y token destino en formato clásico.
token_labels_num = arrayfun(@(k) sprintf('Token %d', k), 1:n_tokens, 'UniformOutput', false);

% Figura adicional: importancia por token origen.
figure('Color','w','Position',[100 100 900 500]);

ax = axes;
ax.Position = [0.12 0.16 0.82 0.70];

bar(ax, 1:n_tokens, imp_tokenO_asca_pct);

xlabel(ax, 'Token origen', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Importancia relativa (%)', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

title(ax, 'ASCA: importancia por token origen', ...
      'FontSize', FS_TITULO, ...
      'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

xticks(ax, 1:n_tokens);
xticklabels(ax, token_labels_num);
xtickangle(ax, 0);

grid(ax, 'on');

exportgraphics(gcf, 'asca_meda_importancia_token_origen.png', 'Resolution', 300);

% Figura adicional: importancia por token destino.
figure('Color','w','Position',[100 100 900 500]);

ax = axes;
ax.Position = [0.12 0.16 0.82 0.70];

bar(ax, 1:n_tokens, imp_tokenD_asca_pct);

xlabel(ax, 'Token destino', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Importancia relativa (%)', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

title(ax, 'ASCA: importancia por token destino', ...
      'FontSize', FS_TITULO, ...
      'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

xticks(ax, 1:n_tokens);
xticklabels(ax, token_labels_num);
xtickangle(ax, 0);

grid(ax, 'on');

exportgraphics(gcf, 'asca_meda_importancia_token_destino.png', 'Resolution', 300);

figure('Color','w','Position',FIG_POS);

bar([imp_tokenO_asca_pct, imp_tokenD_asca_pct, imp_token_comb_asca_pct]);

xticks(1:n_tokens);
xticklabels(token_labels);

xlabel('Token', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel('Importancia relativa (%)', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

legend({'Como origen','Como destino','Combinada'}, ...
       'Location','best', ...
       'FontSize', FS_LEYENDA);

title('ASCA MEDA: importancia relativa por token', ...
      'FontSize', FS_TITULO, ...
      'FontWeight', 'bold');

grid on;
set(gca, 'FontSize', FS_TICKS, ...
         'LineWidth', 1.2);

exportgraphics(gcf, 'asca_meda_importancia_tokens.png', 'Resolution', 300);

%% =========================================================
% 13) MAPA TOKEN ORIGEN - TOKEN DESTINO
% ==========================================================

map_token_effect_asca = accumarray([tokenO_idx tokenD_idx], effect_asca, [n_tokens n_tokens], @sum, 0);
map_token_importance_asca = accumarray([tokenO_idx tokenD_idx], importance_asca, [n_tokens n_tokens], @sum, 0);

figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = [0.12 0.15 0.72 0.70];

imagesc(ax, 1:n_tokens, 1:n_tokens, map_token_effect_asca);

cb = colorbar(ax);
cb.FontSize = FS_TICKS;
cb.Label.String = 'Efecto ASCA: positivo - negativo';
cb.Label.FontSize = FS_EJES;
cb.Label.FontWeight = 'bold';

xticks(ax, 1:n_tokens);
yticks(ax, 1:n_tokens);
xticklabels(ax, token_labels);
yticklabels(ax, token_labels);

xlabel(ax, 'Token destino', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Token origen', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('ASCA MEDA: efecto por pares de tokens', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

exportgraphics(gcf, 'asca_meda_mapa_token_origen_destino_efecto.png', 'Resolution', 300);

figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = [0.12 0.15 0.72 0.70];

imagesc(ax, 1:n_tokens, 1:n_tokens, map_token_importance_asca);

cb = colorbar(ax);
cb.FontSize = FS_TICKS;
cb.Label.String = 'Importancia acumulada ASCA';
cb.Label.FontSize = FS_EJES;
cb.Label.FontWeight = 'bold';

xticks(ax, 1:n_tokens);
yticks(ax, 1:n_tokens);
xticklabels(ax, token_labels);
yticklabels(ax, token_labels);

xlabel(ax, 'Token destino', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Token origen', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('ASCA MEDA: importancia por pares de tokens', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

exportgraphics(gcf, 'asca_meda_mapa_token_origen_destino_importancia.png', 'Resolution', 300);

%% =========================================================
% 14) GUARDAR RESULTADOS COMPLETOS
% ==========================================================

save('asca_results_meda.mat', ...
    'X', ...
    'labels', ...
    'F', ...
    'class_id', ...
    'pair_id', ...
    'tabla_anova_asca', ...
    'parglmo', ...
    'ascao', ...
    'X_factor_clase', ...
    'effect_asca', ...
    'importance_asca', ...
    'importance_sorted', ...
    'idx_sorted_all', ...
    'tabla_variables', ...
    'imp_by_layer_asca', ...
    'imp_by_layer_asca_pct', ...
    'imp_by_head_global_asca', ...
    'imp_by_head_global_asca_pct', ...
    'imp_by_layer_head_asca', ...
    'imp_tokenO_asca', ...
    'imp_tokenD_asca', ...
    'imp_token_comb_asca', ...
    'imp_tokenO_asca_pct', ...
    'imp_tokenD_asca_pct', ...
    'imp_token_comb_asca_pct', ...
    'map_token_effect_asca', ...
    'map_token_importance_asca', ...
    'token_labels');

fprintf('\nASCA con MEDA Toolbox finalizado.\n');
fprintf('Resultados guardados en asca_results_meda.mat\n');
