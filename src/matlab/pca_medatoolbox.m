%% =========================================================
%  PCA sobre la matriz de atención de BERT usando MEDA Toolbox
%  Autor: Adrián Macias Caballero
%
%  Objetivo:
%  - Cargar la matriz X (40 x 9216)
%  - Preprocesar los datos con preprocess2D
%  - Aplicar PCA con pcaEig de MEDA Toolbox
%  - Visualizar scores y varianza explicada
%  - Guardar el modelo PCA para usarlo después en oMEDA
% ==========================================================

clear;
clc;
close all;

%% =========================================================
% 0) AÑADIR MEDA TOOLBOX AL PATH
% ==========================================================

 addpath(genpath('C:\Users\Usuario\Desktop\UNI\Cuarto de Carrera Teleco\doctoradopepe\MEDA Shared Materials-20260301T114916Z-3-001\MEDA Shared Materials\MEDA-Toolbox-master\MEDA-Toolbox-master\toolbox'));
 savepath;

disp('Comprobando funciones de MEDA Toolbox...');

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
% 1) CARGAR DATOS
% ==========================================================
% El archivo debe contener:
%   X               -> matriz 40 x 9216
%   sentences       -> frases
%   true_label      -> etiquetas reales
%   predicted_label -> etiquetas predichas por BERT
%   prob_negative   -> probabilidad clase negativa
%   prob_positive   -> probabilidad clase positiva
% ==========================================================

load('attention_data_40x9216.mat');

disp('Tamaño de la matriz X:');
disp(size(X));

[n_obs, n_vars] = size(X); 
%el numero de observaciones son las filas(frases)=40 
%el numero de variables son las 9216 de la matriz X que son las relaciones
%token a token
fprintf('Número de observaciones (frases): %d\n', n_obs);
fprintf('Número de variables: %d\n', n_vars);

%% =========================================================
% 2) PREPROCESAMIENTO CON MEDA TOOLBOX
% ==========================================================
% Preprocessing:
%   0 -> sin preprocesamiento
%   1 -> centrado en media
%   2 -> autoescalado
%
% En este trabajo usamos autoescalado para que todas las variables
% de atención tengan un tratamiento comparable.
% 1. Se resta su media.
% 2. Se divide por su desviación típica.
% ==========================================================

[Xcs, average, scale] = preprocess2D(X, 'Preprocessing', 2);
%average guarda la media de cada variable
%scale guarda la desviación usada en cada variable
disp('Tamaño de la matriz preprocesada Xcs:');
disp(size(Xcs));

%% =========================================================
% 3) APLICAR PCA CON pcaEig
% ==========================================================
% Como X tiene 40 observaciones y 9216 variables, el número máximo
% efectivo de componentes será como mucho rank(Xcs).
% Al estar centrado, normalmente será 39 = 40 obs - 1 .
% ==========================================================

max_pcs = rank(Xcs); %numero maximo de componentes calculables
pcs_all = 1:max_pcs; %Crea un vector con todas las componentes que se van a calcular. 1:39

fprintf('Número máximo de PCs calculables: %d\n', max_pcs);

model_pca = pcaEig(Xcs, 'PCs', pcs_all);
%la entrada es Xcs que es la matriz autoescalada
%'PCs', pcs_all indicamos que queremos calcular todas las componentes
%posibles

disp('Campos disponibles en model_pca:');
disp(fieldnames(model_pca));

%% =========================================================
% 4) EXTRAER RESULTADOS DEL MODELO PCA
% ==========================================================
% En esta versión de la MEDA Toolbox:
%   model_pca.loads  -> loadings
%   model_pca.scores -> scores
%   model_pca.var    -> varianza total
%   model_pca.lvs    -> componentes consideradas
%   model_pca.type   -> tipo de modelo
% ==========================================================

P_all = model_pca.loads; %extrae los loadings del PCA
%los loadings indican como contribuye cada variable original a cada
%componente principal
T_all = model_pca.scores;
%los scores son las coordenadas de cada frase en el nuevo espacio PCA.
disp('Tipo de modelo:');
disp(model_pca.type);%resultado=PCA

disp('Tamaño de scores T_all:');
disp(size(T_all));%resultado 40(frases)x39(componentes)

disp('Tamaño de loadings P_all:');
disp(size(P_all));%resultado 9216x39

%% =========================================================
% 5) VARIANZA EXPLICADA
% ==========================================================
% La varianza explicada por cada PC(componente) se calcula a partir de la
% suma de cuadrados de los scores.
%
% model_pca.var contiene la varianza total de Xcs.
% sum(T_all.^2,1) contiene la varianza capturada por cada PC.
% ==========================================================

ss_pcs = sum(T_all.^2, 1)';          % suma de cuadrados por PC y el 1 es que suma por columnas ya que son las 39 componentes y se transpone para que quede como vector columna
total_var = model_pca.var;           % varianza total según la toolbox,cantidad total de información/variabilidad que hay en los datos.

explained = 100 * ss_pcs / total_var; %Calcula el porcentaje de varianza explicada por cada componente. ejemplo aqui pc1 explica solo un 15%
cum_explained = cumsum(explained); %Calcula la varianza explicada acumulada. va sumando la varianza que va explicando cada componente para saber cuantas serian necesarias para explicar un alto % de los datos,

disp('Primeras 10 componentes: varianza explicada (%)');
disp(explained(1:min(10,length(explained))));
% Primeras 10 componentes: varianza explicada (%) RESULTADO PUESTO AQUI
%    15.4822
%    10.5118
%     6.6694
%     6.1958
%     5.8219
%     5.3005
%     4.5168
%     4.4450
%     4.2540
%     3.1751
disp('Primeras 10 componentes: varianza explicada acumulada (%)');
disp(cum_explained(1:min(10,length(cum_explained))));
% Primeras 10 componentes: varianza explicada acumulada (%)
%    15.4822
%    25.9941
%    32.6634
%    38.8592
%    44.6812
%    49.9817
%    54.4985
%    58.9435
%    63.1975
%    66.3726

%% =========================================================
% 6) ELEGIR NÚMERO DE COMPONENTES PRINCIPALES
% ==========================================================
% Criterio: conservar las componentes necesarias para alcanzar
% el 95% de varianza explicada acumulada.
% ==========================================================

A = find(cum_explained >= 95, 1, 'first');

fprintf('Número de componentes necesarias para alcanzar el 95%%: %d\n', A);
%resultado = 29, es decir, serian necesarias 29 componentes como minimo
%para explicar el 95% de la varianza.
%% =========================================================
% 7) EXTRAER MODELO REDUCIDO
% ==========================================================

P = P_all(:, 1:A); %Se queda con los loadings de las primeras A componentes.
T = T_all(:, 1:A);%Se queda con los scores de las primeras A componentes.
L = ss_pcs(1:A);%Guarda la suma de cuadrados de las primeras A componentes.
E = explained(1:A);%Guarda la varianza explicada de las primeras A componentes.

disp('Tamaño de P retenido:'); %resultado = 9216x29 
disp(size(P));

disp('Tamaño de T retenido:');%resultado = 40x29
disp(size(T));

%% =========================================================
% 8) GRÁFICO DE VARIANZA EXPLICADA POR COMPONENTE
% ==========================================================

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.84 0.72];

plot(ax, explained, 'o-', ...
    'LineWidth', 1.8, ...
    'MarkerSize', 7);

xlabel(ax, 'Componente principal', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Varianza explicada (%)', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

title(ax, 'PCA MEDA: varianza explicada por componente principal', ...
      'FontSize', FS_TITULO, ...
      'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');

saveas(gcf, 'pca_meda_varianza_explicada.png');

%% =========================================================
% 9) GRÁFICO DE VARIANZA EXPLICADA ACUMULADA
% ==========================================================

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.84 0.72];

plot(ax, cum_explained, 'o-', ...
     'LineWidth', 1.8, ...
     'MarkerSize', 7);

xlabel(ax, 'Número de componentes principales', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Varianza explicada acumulada (%)', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('PCA MEDA: varianza explicada acumulada', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');
hold(ax, 'on');

h80  = yline(ax, 80,  '--r', '80%');
h90  = yline(ax, 90,  '--g', '90%');
h95  = yline(ax, 95,  '--b', '95%');
h100 = yline(ax, 100, '--k', '100%');

h80.FontSize  = FS_LEYENDA;
h90.FontSize  = FS_LEYENDA;
h95.FontSize  = FS_LEYENDA;
h100.FontSize = FS_LEYENDA;

h80.FontWeight  = 'bold';
h90.FontWeight  = 'bold';
h95.FontWeight  = 'bold';
h100.FontWeight = 'bold';

h80.LabelHorizontalAlignment  = 'left';
h90.LabelHorizontalAlignment  = 'left';
h95.LabelHorizontalAlignment  = 'left';
h100.LabelHorizontalAlignment = 'left';

h80.LabelVerticalAlignment  = 'bottom';
h90.LabelVerticalAlignment  = 'bottom';
h95.LabelVerticalAlignment  = 'bottom';
h100.LabelVerticalAlignment = 'bottom';

ylim(ax, [0 105]);

hold(ax, 'off');

saveas(gcf, 'pca_meda_varianza_acumulada.png');

%% =========================================================
% 10) SCORES PC1 VS PC2
% ==========================================================
% Se representan las observaciones en el espacio de las dos primeras
% componentes principales para estudiar si las frases positivas y
% negativas se separan de forma natural.
% ==========================================================

labels = string(true_label(:));%Convierte las etiquetas reales a tipo string.
%positivo
%negativo
figure('Color','w','Position',[100 100 1000 700]);

gscatter(T_all(:,1), T_all(:,2), labels);%scores de PC1,PC2, y labels colorea los puntos según sea positivo/negativo

xlabel(sprintf('PC1 (%.2f%%)', explained(1)), ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(sprintf('PC2 (%.2f%%)', explained(2)), ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

title('PCA MEDA: scores PC1-PC2', ...
      'FontSize', FS_TITULO, ...
      'FontWeight', 'bold');

set(gca, 'FontSize', FS_TICKS, ...
         'LineWidth', 1.2);

grid on;

saveas(gcf, 'pca_meda_scores_pc1_pc2.png');

%% =========================================================
% 11) SCORES PC1 VS PC3
% ==========================================================

if size(T_all,2) >= 3 %comprobar que existen 3 PC

    figure('Color','w','Position',[100 100 1000 700]);

    gscatter(T_all(:,1), T_all(:,3), labels);

    xlabel(sprintf('PC1 (%.2f%%)', explained(1)), ...
           'FontSize', FS_EJES, ...
           'FontWeight', 'bold');

    ylabel(sprintf('PC3 (%.2f%%)', explained(3)), ...
           'FontSize', FS_EJES, ...
           'FontWeight', 'bold');

    title('PCA MEDA: scores PC1-PC3', ...
          'FontSize', FS_TITULO, ...
          'FontWeight', 'bold');

    set(gca, 'FontSize', FS_TICKS, ...
             'LineWidth', 1.2);

    grid on;

    saveas(gcf, 'pca_meda_scores_pc1_pc3.png');
end

%% =========================================================
% 12) FIGURA DE SCORES USANDO FUNCIÓN scores DE MEDA TOOLBOX
% ==========================================================
% Esta parte usa directamente la visualización de la toolbox.
% Se crea un modelo reducido solo con PC1 y PC2 para evitar que
% scores genere todas las combinaciones de componentes.
% ==========================================================

labels_num = grp2idx(categorical(labels));

model_scores_12 = model_pca;
model_scores_12.lvs = [1 2]; %lvs significa variables latentes = componentes
model_scores_12.loads = P_all(:,1:2);
model_scores_12.scores = T_all(:,1:2);
model_scores_12.var = total_var;

try
    scores(model_scores_12, ...
        'ObsClass', labels_num, ...
        'ObsLabel', (1:n_obs)', ...
        'Title', 'PCA MEDA Toolbox: PC1-PC2', ...
        'BlurIndex', Inf);
catch ME
    warning('No se pudo generar la figura con scores de MEDA Toolbox: %s', ME.message);
end

%% =========================================================
% 13) MOSTRAR ALGUNOS RESULTADOS
% ==========================================================

disp('Primeras 5 filas de T_all:');
disp(T_all(1:min(5,size(T_all,1)), 1:min(5,size(T_all,2))));

disp('Primeras 5 filas y 5 columnas de P_all:');
disp(P_all(1:min(5,size(P_all,1)), 1:min(5,size(P_all,2))));

%% =========================================================
% 14) GUARDAR RESULTADOS DEL PCA MEDA
% ==========================================================

save('pca_attention_results_meda.mat', ...
    'X', 'Xcs', ...
    'average', 'scale', ...
    'model_pca', ...
    'P_all', 'T_all', ...
    'ss_pcs', 'total_var', ...
    'explained', 'cum_explained', ...
    'A', 'P', 'T', 'L', 'E', ...
    'sentences', 'true_label', 'predicted_label', ...
    'prob_negative', 'prob_positive');

disp('Archivo guardado: pca_attention_results_meda.mat');

%% =========================================================
% 15) RESUMEN FINAL
% ==========================================================

fprintf('\nResumen del PCA con MEDA Toolbox:\n');
fprintf(' - Observaciones: %d\n', n_obs);
fprintf(' - Variables: %d\n', n_vars);
fprintf(' - Componentes calculadas: %d\n', max_pcs);
fprintf(' - Componentes retenidas para 95%%: %d\n', A);
fprintf(' - Varianza explicada acumulada con %d componentes: %.2f%%\n', ...
    A, cum_explained(A));