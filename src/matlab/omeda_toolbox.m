%% =========================================================
%  oMEDA sobre la matriz de atención de BERT usando MEDA Toolbox
%  Autor: Adrián Macias Caballero
%
%  Objetivo:
%  - Cargar el PCA calculado previamente con MEDA Toolbox
%  - Construir un contraste positivo vs negativo mediante dummy
%  - Calcular oMEDA usando la función omeda de MEDA Toolbox
%  - Analizar la importancia de las variables de atención
%  - Agrupar la importancia por capa, cabeza y token
% ==========================================================

clear;
clc;
close all;

%% =========================================================
% 0) AÑADIR MEDA TOOLBOX AL PATH
% ==========================================================
% Cambia esta ruta si hace falta.
% Si MATLAB ya encuentra omeda, pcaEig y preprocess2D, puedes dejarlo comentado.
% ==========================================================

addpath(genpath('C:\Users\Usuario\Desktop\UNI\Cuarto de Carrera Teleco\doctoradopepe\MEDA Shared Materials-20260301T114916Z-3-001\MEDA Shared Materials\MEDA-Toolbox-master\MEDA-Toolbox-master\toolbox'));
savepath;

disp('Comprobando funciones de MEDA Toolbox...');

if isempty(which('omeda'))
    error('No se encuentra la función omeda. Añade la MEDA Toolbox al path de MATLAB.');
end

if isempty(which('pcaEig'))
    error('No se encuentra pcaEig. Añade la MEDA Toolbox al path de MATLAB.');
end

if isempty(which('preprocess2D'))
    error('No se encuentra preprocess2D. Añade la MEDA Toolbox al path de MATLAB.');
end

disp('MEDA Toolbox encontrada correctamente.');

%% =========================================================
% Tamaños de fuente para las figuras
% ==========================================================

FS_TITULO  = 22;
FS_EJES    = 18;
FS_TICKS   = 15;
FS_LEYENDA = 16;

%% =========================================================
% 1) CARGAR RESULTADOS DEL PCA MEDA
% ==========================================================
% Este archivo viene del script PCA con MEDA Toolbox.
%
% Debe contener:
%   X              -> matriz original 40 x 9216
%   Xcs            -> matriz preprocesada con preprocess2D
%   average        -> media usada en el preprocesamiento
%   scale          -> escala usada en el preprocesamiento
%   model_pca      -> modelo PCA generado con pcaEig
%   P_all          -> loadings de todas las PCs
%   T_all          -> scores de todas las PCs
%   A              -> número de PCs retenidas para 95%
%   P              -> loadings retenidos
%   T              -> scores retenidos
%   true_label     -> etiquetas reales
% ==========================================================

load('pca_attention_results_meda.mat');

disp('Archivo PCA MEDA cargado correctamente.');

[n_obs, n_vars] = size(Xcs);

fprintf('Número de observaciones: %d\n', n_obs); %resultado como antes = 40 frases
fprintf('Número de variables: %d\n', n_vars); %resultado como antes = 9216 variables
fprintf('Número de componentes retenidas A: %d\n', A); %resultado de antes A = 29 componentes que explicaban el 95% y es el modelo pca reducido

%% =========================================================
% 2) PREPARAR ETIQUETAS
% ==========================================================
% Convertimos las etiquetas a string para trabajar cómodamente.
% Deben existir dos grupos:
%   positive
%   negative
% ==========================================================

labels = string(true_label(:)); %pone las etiquetas en positivo y negativo modo string

idx_pos = find(labels == "positive");
idx_neg = find(labels == "negative");

fprintf('\nNúmero de frases positivas: %d\n', numel(idx_pos));
fprintf('Número de frases negativas: %d\n', numel(idx_neg));

if isempty(idx_pos) || isempty(idx_neg)
    error('No se han encontrado correctamente las clases positive y negative.');
end

%% =========================================================
% 3) CONSTRUIR VARIABLE DUMMY PARA oMEDA
% ==========================================================
% La función omeda de la MEDA Toolbox compara observaciones mediante
% una variable dummy.
%
% En este caso:
%   +1 -> frases positivas
%   -1 -> frases negativas
%
% De esta forma, el vector oMEDA representa el contraste:
%   positive - negative
%
% Interpretación:
%   omeda_vec > 0  -> variable asociada al grupo positivo
%   omeda_vec < 0  -> variable asociada al grupo negativo
%   omeda_vec ≈ 0  -> variable con contribución baja al contraste
% ==========================================================

dummy = zeros(n_obs, 1);%Crea un vector columna de ceros con tantas filas como observaciones.
dummy(idx_pos) =  1; %Asigna valor +1 a las frases positivas.
dummy(idx_neg) = -1; %Asigna valor -1 a las frases negativas.

disp('Vector dummy construido:');%vector que define el contraste positivo frente a negativo.
disp(table((1:n_obs)', labels, dummy, ...
    'VariableNames', {'Observacion','Clase','Dummy'}));

%% =========================================================
% 4) SELECCIONAR LOADINGS DEL MODELO PCA MEDA
% ==========================================================
% P contiene los loadings retenidos del PCA realizado con pcaEig.
%
% Si quieres usar todas las PCs calculadas, puedes sustituir:
%   P_omeda = P;
% por:
%   P_omeda = P_all;
%
% En este trabajo se usa el modelo reducido que alcanza el 95% de
% varianza explicada acumulada, guardado previamente como P.
% ==========================================================

P_omeda = P; %los loadings eran las variables por tanto tiene el size de 9216 x 29
%contiene la relación entre las componentes principales y las variables originales.
disp('Tamaño de P_omeda:');
disp(size(P_omeda));

%% =========================================================
% 5) CALCULAR oMEDA CON MEDA TOOLBOX
% ==========================================================
% Esta es la línea principal.
%
% Xcs      -> matriz preprocesada con preprocess2D
% dummy    -> contraste positivo vs negativo
% P_omeda  -> loadings del PCA calculado con pcaEig
%
% La función omeda proyecta el contraste definido por dummy
% sobre el espacio original de variables.
% ==========================================================

omeda_vec = omeda(Xcs, dummy, P_omeda); %tamaño de 9216 variables x 1

disp('Tamaño del vector oMEDA:');
disp(size(omeda_vec));

if size(omeda_vec,1) ~= n_vars
    error('El vector oMEDA no tiene el tamaño esperado de 9216 variables.');
end

%% =========================================================
% 6) GUARDAR VECTOR oMEDA BÁSICO
% ==========================================================

tabla_omeda = table((1:n_vars)', omeda_vec, abs(omeda_vec), ...
    'VariableNames', {'Variable','oMEDA','Abs_oMEDA'});

writetable(tabla_omeda, 'omeda_meda_vector_variables.csv');

disp('Archivo guardado: omeda_meda_vector_variables.csv');

%% =========================================================
% 7) GRÁFICO oMEDA GLOBAL
% ==========================================================

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.84 0.72];

plot(ax, omeda_vec, 'LineWidth', 1.4);

xlabel(ax, 'Variable de atención', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Valor oMEDA', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('oMEDA MEDA Toolbox: positivo - negativo', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');

exportgraphics(gcf, 'omeda_meda_positivo_negativo.png', 'Resolution', 300);

%% =========================================================
% 8) IMPORTANCIA DE TODAS LAS VARIABLES
% ==========================================================
% Se toma el valor absoluto de oMEDA para medir la importancia
% independientemente del signo.
%
% El signo mantiene la dirección del contraste:
%   positivo -> más asociado a frases positivas
%   negativo -> más asociado a frases negativas
% ==========================================================

importance = abs(omeda_vec);%para importancia solo interesa cuánto pesa la variable, no si está asociada a positiva o negativa.

[importance_sorted, idx_sorted_all] = sort(importance, 'descend');%Ordena las variables de mayor a menor importancia.

K = 50;%Define cuántas variables importantes quieres mostrar.
topK_idx = idx_sorted_all(1:K);%Extrae los índices de las 50 variables más importantes.
topK_vals = omeda_vec(topK_idx);%Extrae los valores oMEDA con signo de esas 50 variables.
topK_abs  = importance(topK_idx);%Extrae los valores absolutos de esas 50 variables.

tabla_topK = table(topK_idx, topK_vals, topK_abs, ...
    'VariableNames', {'Variable','oMEDA','Abs_oMEDA'});

fprintf('\nTop %d variables según |oMEDA|:\n', K);
disp(tabla_topK);

writetable(tabla_topK, 'omeda_meda_top50_variables.csv');

%% =========================================================
% 9) SELECCIÓN AUTOMÁTICA DE VARIABLES RELEVANTES
% ==========================================================
% Usamos un umbral estadístico:
%   media(|oMEDA|) + 2 * desviación típica(|oMEDA|)
%
% Así no se elige un número de variables arbitrario.
% ==========================================================

thr = mean(importance) + 2*std(importance);%Calcula un umbral estadístico.

relevant_idx = find(importance >= thr);%Busca las variables cuya importancia supera ese umbral.
relevant_vals = omeda_vec(relevant_idx);%Extrae sus valores oMEDA con signo.
relevant_abs  = importance(relevant_idx);%Extrae sus valores absolutos.

fprintf('\nVariables relevantes según umbral |oMEDA| >= media + 2sigma: %d\n', numel(relevant_idx));
%resultado = 520 variables superan el umbral
tabla_relevantes = table(relevant_idx, relevant_vals, relevant_abs, ...
    'VariableNames', {'Variable','oMEDA','Abs_oMEDA'});

writetable(tabla_relevantes, 'omeda_meda_variables_relevantes.csv');

%% =========================================================
% 10) HISTOGRAMA DE IMPORTANCIAS
% ==========================================================

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.84 0.72];

histogram(ax, importance, 50);%El 50 indica el número de barras o intervalos.

xlabel(ax, '|oMEDA|', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Frecuencia', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('Distribución de la importancia de las 9216 variables', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');
hold(ax, 'on');

hthr = xline(ax, thr, '--r', 'Umbral \mu+2\sigma');

hthr.FontSize = FS_LEYENDA;
hthr.FontWeight = 'bold';
hthr.LabelVerticalAlignment = 'top';
hthr.LabelHorizontalAlignment = 'left';

hold(ax, 'off');

exportgraphics(gcf, 'omeda_meda_distribucion_importancia.png', 'Resolution', 300);

%% =========================================================
% 11) AGRUPAR IMPORTANCIA POR CAPA Y HEAD
% ==========================================================
% Cada variable de atención se corresponde con:
%
%   layer, head, token origen, token destino
%
% La vectorización original seguía este orden:
%   layer -> head -> matriz 8x8 aplanada fila por fila
%
% Por tanto:
%   9216 = 12 layers * 12 heads * 8 tokens origen * 8 tokens destino
% ==========================================================

n_layers = 12;
n_heads  = 12;
n_tokens = 8;

if n_vars ~= n_layers*n_heads*n_tokens*n_tokens
    error('El número de variables no coincide con 12*12*8*8 = 9216.');
end

imp_by_layer = zeros(n_layers,1);%Crea un vector para acumular importancia por capa.Tendrá 12 valores, uno por capa.
imp_by_head  = zeros(n_layers,n_heads);%Crea una matriz para acumular importancia por capa y cabeza.

for var_idx = 1:n_vars%Recorre las 9216 variables de atención.

    zero_idx = var_idx - 1;

    layer_idx = floor(zero_idx / (n_heads*n_tokens*n_tokens));%Calcula a qué capa pertenece la variable.Cada capa contiene 768 vars.
    rem1      = mod(zero_idx, n_heads*n_tokens*n_tokens);%Calcula el resto dentro de esa capa.

    head_idx  = floor(rem1 / (n_tokens*n_tokens));%Calcula a qué cabeza pertenece la variable.Cada cabeza contiene 64 vars.

    imp_by_layer(layer_idx+1) = imp_by_layer(layer_idx+1) + importance(var_idx);%Suma la importancia de esa variable a la capa correspondiente.
    imp_by_head(layer_idx+1, head_idx+1) = imp_by_head(layer_idx+1, head_idx+1) + importance(var_idx);%Suma la importancia a la combinación capa-cabeza correspondiente.
end

tabla_layer = table((1:n_layers)', imp_by_layer, ...
    100*imp_by_layer/sum(imp_by_layer), ...
    'VariableNames', {'Layer','Importancia','Importancia_pct'});

writetable(tabla_layer, 'omeda_meda_importancia_layer.csv');

%% =========================================================
% 12) IMPORTANCIA POR CAPA
% ==========================================================

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.84 0.72];

bar(ax, imp_by_layer);

xlabel(ax, 'Layer', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Importancia acumulada (|oMEDA|)', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('oMEDA MEDA Toolbox: importancia por capa', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

xticks(ax, 1:n_layers);
grid(ax, 'on');

exportgraphics(gcf, 'omeda_meda_importancia_por_capa.png', 'Resolution', 300);

%% =========================================================
% 13) HEATMAP DE IMPORTANCIA POR CAPA Y HEAD
% ==========================================================

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.78 0.72];

imagesc(ax, 1:n_heads, 1:n_layers, imp_by_head);

cb = colorbar(ax);
cb.FontSize = FS_TICKS;
cb.Label.String = 'Importancia acumulada (|oMEDA|)';
cb.Label.FontSize = FS_EJES;
cb.Label.FontWeight = 'bold';

xlabel(ax, 'Head', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Layer', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('oMEDA MEDA Toolbox: importancia por layer-head', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'YDir', 'reverse', ...
        'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

xticks(ax, 1:n_heads);
yticks(ax, 1:n_layers);

grid(ax, 'off');

exportgraphics(gcf, 'omeda_meda_importancia_layer_head.png', 'Resolution', 300);

%% =========================================================
% 14) IMPORTANCIA POR TOKEN EN oMEDA
% ==========================================================
% Para obtener la importancia por token:
%
%   1) Se suma |oMEDA| cuando el token actúa como origen.
%   2) Se suma |oMEDA| cuando el token actúa como destino.
%   3) Se combinan ambas contribuciones.
%
% Esto permite estudiar qué posiciones de la frase tienen mayor peso
% en el contraste positivo vs negativo.
% ==========================================================

token_labels = ["[CLS]"; "Token 2"; "Token 3"; "Token 4"; ...
                "Token 5"; "Token 6"; "."; "[SEP]"];%Define las etiquetas de las 8 posiciones de token.

imp_tokenO_omeda = zeros(n_tokens,1);%Crea vector para acumular importancia cuando el token actúa como origen.
imp_tokenD_omeda = zeros(n_tokens,1);%Crea vector para acumular importancia cuando el token actúa como destino.

for var_idx = 1:n_vars

    zero_idx = var_idx - 1;

    layer_idx = floor(zero_idx / (n_heads*n_tokens*n_tokens));
    rem1      = mod(zero_idx, n_heads*n_tokens*n_tokens);

    head_idx  = floor(rem1 / (n_tokens*n_tokens));
    rem2      = mod(rem1, n_tokens*n_tokens);

    tokenO_idx = floor(rem2 / n_tokens) + 1;
    tokenD_idx = mod(rem2, n_tokens) + 1;

    imp_tokenO_omeda(tokenO_idx) = imp_tokenO_omeda(tokenO_idx) + importance(var_idx);
    imp_tokenD_omeda(tokenD_idx) = imp_tokenD_omeda(tokenD_idx) + importance(var_idx);
end

imp_token_comb_omeda = imp_tokenO_omeda + imp_tokenD_omeda;

imp_tokenO_omeda_pct     = 100 * imp_tokenO_omeda     / sum(imp_tokenO_omeda);
imp_tokenD_omeda_pct     = 100 * imp_tokenD_omeda     / sum(imp_tokenD_omeda);
imp_token_comb_omeda_pct = 100 * imp_token_comb_omeda / sum(imp_token_comb_omeda);

tabla_omeda_tokens = table(token_labels, ...
    imp_tokenO_omeda_pct, ...
    imp_tokenD_omeda_pct, ...
    imp_token_comb_omeda_pct, ...
    'VariableNames', {'Token','Origen','Destino','Combinada'});

writetable(tabla_omeda_tokens, 'omeda_meda_importancia_tokens.csv');

%% =========================================================
% 15) FIGURA IMPORTANCIA POR TOKEN
% ==========================================================

figure('Color','w','Position',[100 100 1200 700]);

bar([imp_tokenO_omeda_pct, imp_tokenD_omeda_pct, imp_token_comb_omeda_pct]);

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

title('oMEDA MEDA Toolbox: importancia relativa por token', ...
      'FontSize', FS_TITULO, ...
      'FontWeight', 'bold');

grid on;

set(gca, 'FontSize', FS_TICKS, ...
         'LineWidth', 1.2);

exportgraphics(gcf, 'omeda_meda_importancia_tokens.png', 'Resolution', 300);

%% =========================================================
% 16) GUARDAR RESULTADOS COMPLETOS
% ==========================================================

save('omeda_attention_results_meda.mat', ...
    'omeda_vec', ...
    'importance', ...
    'importance_sorted', ...
    'idx_sorted_all', ...
    'thr', ...
    'relevant_idx', ...
    'relevant_vals', ...
    'imp_by_layer', ...
    'imp_by_head', ...
    'imp_tokenO_omeda', ...
    'imp_tokenD_omeda', ...
    'imp_token_comb_omeda', ...
    'imp_tokenO_omeda_pct', ...
    'imp_tokenD_omeda_pct', ...
    'imp_token_comb_omeda_pct', ...
    'dummy', ...
    'labels', ...
    'P_omeda', ...
    'A');

disp('Archivo guardado: omeda_attention_results_meda.mat');

%% =========================================================
% 17) RESUMEN FINAL
% ==========================================================

fprintf('\nResumen oMEDA MEDA Toolbox:\n');
fprintf(' - Observaciones: %d\n', n_obs);
fprintf(' - Variables: %d\n', n_vars);
fprintf(' - Componentes PCA usadas: %d\n', size(P_omeda,2));
fprintf(' - Variables relevantes según umbral: %d\n', numel(relevant_idx));
fprintf(' - Máximo |oMEDA|: %.6f\n', max(importance));
fprintf(' - Media |oMEDA|: %.6f\n', mean(importance));
fprintf(' - Umbral media + 2sigma: %.6f\n', thr);