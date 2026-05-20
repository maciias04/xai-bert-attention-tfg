%% =========================================================
%  PCA sobre la matriz de atención de BERT
%  Autor: Adrián Macias Caballero
%  Objetivo:
%  - Cargar la matriz X (40 x 9216)
%  - Estandarizar los datos
%  - Aplicar PCA
%  - Visualizar resultados
%  - Guardar el modelo PCA para usarlo después en oMEDA
% ==========================================================

clear;
clc;
close all;
% Tamaños de fuente para las figuras
FS_TITULO  = 22;
FS_EJES    = 18;
FS_TICKS   = 15;
FS_LEYENDA = 16;
%% =========================================================
% 1) CARGAR DATOS
% ==========================================================
% Este archivo .mat viene del script de Python.
% Debe contener como mínimo:
%   X               -> matriz 40 x 9216
%   sentences       -> frases
%   true_label      -> etiquetas reales
%   predicted_label -> etiquetas predichas por BERT
%   prob_negative   -> probabilidad clase negativa
%   prob_positive   -> probabilidad clase positiva
% ==========================================================

load('attention_data_40x9216.mat');

%% =========================================================
% 2) COMPROBAR TAMAÑO DE LA MATRIZ
% ==========================================================
% En PCA, MATLAB espera:
%   filas    = observaciones
%   columnas = variables
%
% En tu caso:
%   40 filas    = 40 frases
%   9216 columnas = 9216 variables de atención
% ==========================================================

disp('Tamaño de la matriz X:');
disp(size(X));

[n_obs, n_vars] = size(X);

fprintf('Número de observaciones (frases): %d\n', n_obs);
fprintf('Número de variables: %d\n', n_vars);

%% =========================================================
% 3) ESTANDARIZAR LA MATRIZ
% ==========================================================
% zscore(X) centra cada columna en media 0
% y la escala a desviación típica 1.
%
% Esto suele ser recomendable antes de PCA para que todas las
% variables tengan un tratamiento comparable.
% ==========================================================

Xz = zscore(X);

disp('Tamaño de la matriz estandarizada Xz:');
disp(size(Xz));

%% =========================================================
% 4) APLICAR PCA
% ==========================================================
% Salidas principales:
%   coeff     -> loadings (cargas)
%   score     -> scores (proyección de las observaciones)
%   latent    -> autovalores / varianza de cada componente
%   tsquared  -> estadístico Hotelling T^2
%   explained -> porcentaje de varianza explicada por cada PC
%   mu        -> media usada internamente por PCA
%
% Como ya hemos hecho zscore(X), aplicamos PCA sobre Xz.
% ==========================================================

[coeff, score, latent, tsquared, explained, mu] = pca(Xz);

%% =========================================================
% 5) CALCULAR VARIANZA EXPLICADA ACUMULADA
% ==========================================================
% explained(i)     = % de varianza explicada por la componente i
% cumsum(explained)= % acumulado
% ==========================================================

cum_explained = cumsum(explained);

disp('Primeras 10 componentes: varianza explicada (%)');
disp(explained(1:min(10,length(explained))));

disp('Primeras 10 componentes: varianza explicada acumulada (%)');
disp(cum_explained(1:min(10,length(cum_explained))));

%% =========================================================
% 6) ELEGIR NÚMERO DE COMPONENTES PRINCIPALES
% ==========================================================
% Aquí tomamos como criterio el 95% de varianza explicada.
% Puedes cambiar 95 por 90, 85, etc. según lo que te convenga.
% ==========================================================

A = find(cum_explained >= 95, 1, 'first');

fprintf('Número de componentes necesarias para alcanzar el 95%%: %d\n', A);

%% =========================================================
% 7) EXTRAER EL MODELO REDUCIDO
% ==========================================================
% P -> loadings retenidos
% T -> scores retenidos
% L -> autovalores retenidos
% E -> varianza explicada retenida
% ==========================================================

P = coeff(:, 1:A);
T = score(:, 1:A);
L = latent(1:A);
E = explained(1:A);

disp('Tamaño de P (loadings retenidos):');
disp(size(P));

disp('Tamaño de T (scores retenidos):');
disp(size(T));

%% =========================================================
% 8) GRÁFICO DE VARIANZA EXPLICADA POR COMPONENTE
% ==========================================================
figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.84 0.72]; 
% [izquierda abajo ancho alto]
% baja el alto para dejar hueco al título

plot(ax, explained, 'o-', 'LineWidth', 1.8, 'MarkerSize', 7);

xlabel(ax, 'Componente principal', ...
       'FontSize', 18, 'FontWeight', 'bold');

ylabel(ax, 'Varianza explicada (%)', ...
       'FontSize', 18, 'FontWeight', 'bold');

title(ax, 'Varianza explicada por cada componente principal', ...
      'FontSize', 22, 'FontWeight', 'bold');

set(ax, 'FontSize', 15, 'LineWidth', 1.2);

grid(ax, 'on');
%% =========================================================
% 9) GRÁFICO DE VARIANZA EXPLICADA ACUMULADA
% ==========================================================
% Muy útil para decidir cuántas componentes conservar.
% ==========================================================

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.84 0.72]; 
% [izquierda abajo ancho alto]
% Se deja margen superior para que salga bien el título

plot(ax, cum_explained, 'o-', ...
     'LineWidth', 1.8, ...
     'MarkerSize', 7);

xlabel(ax, 'Número de componentes principales', ...
       'FontSize', 18, ...
       'FontWeight', 'bold');

ylabel(ax, 'Varianza explicada acumulada (%)', ...
       'FontSize', 18, ...
       'FontWeight', 'bold');

sgtitle('Varianza explicada acumulada', ...
        'FontSize', 22, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', 15, ...
        'LineWidth', 1.2);

grid(ax, 'on');
hold(ax, 'on');

% Barras de referencia
h80  = yline(ax, 80,  '--r', '80%');
h90  = yline(ax, 90,  '--g', '90%');
h100 = yline(ax, 100, '--k', '100%');

% Tamaño de las etiquetas de las barras
h80.FontSize  = 16;
h90.FontSize  = 16;
h100.FontSize = 16;

h80.FontWeight  = 'bold';
h90.FontWeight  = 'bold';
h100.FontWeight = 'bold';

% Posición de las etiquetas
h80.LabelHorizontalAlignment  = 'left';
h90.LabelHorizontalAlignment  = 'left';
h100.LabelHorizontalAlignment = 'left';

h80.LabelVerticalAlignment  = 'bottom';
h90.LabelVerticalAlignment  = 'bottom';
h100.LabelVerticalAlignment = 'bottom';

% Límites para que se vea bien la línea del 100%
ylim(ax, [0 105]);

hold(ax, 'off');
%% =========================================================
% 10) REPRESENTAR LOS SCORES EN PC1 VS PC2
% ==========================================================
% Esto sirve para ver si las frases positivas y negativas
% se separan en el espacio PCA.
%
% Convertimos etiquetas a string por si MATLAB las carga
% como celdas o formato no directamente compatible.
% ==========================================================

labels = string(true_label(:));

figure;
gscatter(score(:,1), score(:,2), labels);
xlabel('PC1');
ylabel('PC2');
title('Representación de las observaciones en PC1-PC2');
grid on;

%% =========================================================
% 11) REPRESENTAR LOS SCORES EN PC1 VS PC3 (OPCIONAL)
% ==========================================================
% Puede ser útil si PC2 no separa bien las clases.
% ==========================================================

if size(score,2) >= 3
    figure;
    gscatter(score(:,1), score(:,3), labels);
    xlabel('PC1');
    ylabel('PC3');
    title('Representación de las observaciones en PC1-PC3');
    grid on;
end

%% =========================================================
% 12) MOSTRAR ALGUNOS RESULTADOS IMPORTANTES
% ==========================================================
% Esto ayuda a inspeccionar los primeros scores.
% ==========================================================

disp('Primeras 5 filas de T (scores retenidos):');
disp(T(1:min(5,size(T,1)), :));

disp('Primeras 5 filas y 5 columnas de P (loadings retenidos):');
disp(P(1:min(5,size(P,1)), 1:min(5,size(P,2))));

%% =========================================================
% 13) GUARDAR RESULTADOS DEL PCA
% ==========================================================
% Guardamos todo lo importante para usarlo después en oMEDA
% sin tener que recalcular el PCA otra vez.
% ==========================================================

save('pca_attention_results.mat', ...
    'X', 'Xz', ...
    'coeff', 'score', 'latent', 'tsquared', 'explained', 'mu', ...
    'cum_explained', 'A', ...
    'P', 'T', 'L', 'E', ...
    'sentences', 'true_label', 'predicted_label', ...
    'prob_negative', 'prob_positive');

disp('Archivo guardado: pca_attention_results.mat');

%% =========================================================
% 14) RESUMEN FINAL POR PANTALLA
% ==========================================================

fprintf('\nResumen del PCA:\n');
fprintf(' - Observaciones: %d\n', n_obs);
fprintf(' - Variables: %d\n', n_vars);
fprintf(' - Componentes retenidas (95%%): %d\n', A);
fprintf(' - Varianza explicada acumulada con %d componentes: %.2f%%\n', ...
    A, cum_explained(A));