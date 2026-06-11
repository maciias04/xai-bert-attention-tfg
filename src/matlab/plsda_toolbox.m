%% =========================================================
%  PLS-DA sobre la matriz de atención de BERT usando MEDA Toolbox
%  Autor: Adrián Macias Caballero
%
%  Objetivo:
%  - Repetir el análisis multivariante usando PLS-DA
%  - Usar únicamente funciones de la MEDA Toolbox para el modelo PLS
%  - Mantener coherencia con PCA y oMEDA calculados con MEDA Toolbox
%  - Obtener scores, matriz de confusión e importancia por capa/head/token
% ==========================================================

clear;
clc;
close all;

%% =========================================================
% 0) AÑADIR MEDA TOOLBOX AL PATH
% ==========================================================
% Cambia esta ruta si hace falta.
% Si MATLAB ya encuentra crossvalPlsDA, simpls y preprocess2D,
% puedes dejar estas dos líneas comentadas.
% ==========================================================

addpath(genpath('C:\Users\Usuario\Desktop\UNI\Cuarto de Carrera Teleco\doctoradopepe\MEDA Shared Materials-20260301T114916Z-3-001\MEDA Shared Materials\MEDA-Toolbox-master\MEDA-Toolbox-master\toolbox'));
savepath;

disp('Comprobando funciones de MEDA Toolbox...');

if isempty(which('crossvalPlsDA'))
    error('No se encuentra crossvalPlsDA. Añade la MEDA Toolbox al path de MATLAB.');
end

if isempty(which('simpls'))
    error('No se encuentra simpls. Añade la MEDA Toolbox al path de MATLAB.');
end

if isempty(which('preprocess2D'))
    error('No se encuentra preprocess2D. Añade la MEDA Toolbox al path de MATLAB.');
end

if isempty(which('varPls'))
    warning('No se encuentra varPls. Se omitirá el cálculo de varianza capturada.');
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
% 2) CARGAR DATOS DEL PCA MEDA
% ==========================================================
% Se carga el archivo generado por el script PCA con MEDA Toolbox.
%
% Debe contener:
%   X              -> matriz original 40 x 9216
%   Xcs            -> matriz preprocesada con preprocess2D
%   average        -> media usada en PCA
%   scale          -> escala usada en PCA
%   true_label     -> etiquetas reales
%   predicted_label
%   prob_negative
%   prob_positive
% ==========================================================

load('pca_attention_results_meda.mat');

if ~exist('X', 'var')
    error('No se encuentra la matriz X en pca_attention_results_meda.mat.');
end

if ~exist('Xcs', 'var')
    error('No se encuentra la matriz Xcs. Ejecuta antes el PCA con MEDA Toolbox.');
end

labels = string(true_label(:));%Convierte las etiquetas reales a formato string.

X = double(X);
Xcs = double(Xcs);

[n_obs, n_vars] = size(X);%como siempre n_obs = filas = 40 frases y n_vars = columnas = 9216

fprintf('Número de observaciones: %d\n', n_obs);
fprintf('Número de variables: %d\n', n_vars);

disp('Etiquetas encontradas:');
disp(unique(labels));

%% =========================================================
% 3) CODIFICAR ETIQUETAS PARA PLS-DA
% ==========================================================
% La función crossvalPlsDA de MEDA Toolbox espera variables dummy:
%
%   positive -> +1
%   negative -> -1
%
% Esta codificación mantiene coherencia con el oMEDA anterior,
% donde el contraste era positivo - negativo.
% ==========================================================

Y = zeros(n_obs, 1);%Crea un vector columna de ceros de tamaño 40 × 1. Este vector será la variable respuesta de PLS-DA.
Y(labels == "positive") =  1;%asigna valor +1 a las frases positivas
Y(labels == "negative") = -1;%asigna valor -1 a las frases negativas

if any(Y == 0)
    error('Hay observaciones sin etiqueta positive/negative correctamente asignada.');
end

fprintf('Frases positivas: %d\n', sum(Y == 1));%resultado = 20
fprintf('Frases negativas: %d\n', sum(Y == -1));%resultado = 20

%% =========================================================
% 4) VALIDACIÓN CRUZADA PLS-DA CON MEDA TOOLBOX
% ==========================================================
% crossvalPlsDA realiza validación cruzada k-fold por filas.
% Se usa AUC como criterio de selección del número de LVs.
%
% Como hay 20 frases por clase, MaxBlock = 20 equivale a una
% validación muy exigente, dejando una observación de cada clase
% por bloque aproximadamente.
% ==========================================================

maxLV = min([10, rank(Xcs), n_obs-1]);%Define el número máximo de variables latentes que se van a probar.
%se limita a 10 para no hacer un modelo demasiado complejo. aunque el
%maximo seria 39 = n_obs - 1
lvs_candidates = 1:maxLV;%Crea el vector de variables latentes candidatas.

maxBlock = min(sum(Y == 1), sum(Y == -1));%Define el número de bloques para validación cruzada.

fprintf('\nNúmero máximo de LVs evaluadas: %d\n', maxLV); %resultado = 10
fprintf('Número de bloques de validación cruzada: %d\n', maxBlock); %resultado = 20

[AUC_raw, nze] = crossvalPlsDA(X, Y, ... 
    'LVs', lvs_candidates, ... %Indica qué variables latentes se van a probar.
    'VarNumber', n_vars, ... %Indica el número de variables consideradas.
    'MaxBlock', maxBlock, ... %Indica el número de bloques de validación cruzada.
    'PreprocessingX', 2, ... %Indica que X debe autoescalarse dentro de la validación cruzada.
    'PreprocessingY', 0, ...%Indica que Y no se preprocesa.porque Y ya es una variable dummy +1/-1.
    'Selection', 'SR', ... %Indica el criterio interno de selección usado por la toolbox.suele referirse a Selectivity Ratio.
    'Plot', false);
%Esta función realiza validación cruzada para PLS-DA
% crossvalPlsDA puede devolver AUC como matriz 2D o 3D.
% Para una sola respuesta Y, se reduce a un vector por LV.
if ndims(AUC_raw) == 3
    auc_cv = squeeze(AUC_raw(:,1,1));%Si viene en 3D, se reduce a un vector.
else
    auc_cv = AUC_raw(:,1);%Si viene en 2D, se toma la primera columna.
end

auc_cv_clean = auc_cv;%copia de auc_cv
auc_cv_clean(isnan(auc_cv_clean)) = -Inf;

[best_auc, idx_best] = max(auc_cv_clean);%Busca el AUC máximo.best_auc es el valor máximo.idx_best es la posición donde aparece.
bestLV = lvs_candidates(idx_best);%Convierte la posición del máximo AUC en número de variables latentes.

fprintf('\nNúmero óptimo de variables latentes seleccionado: %d\n', bestLV);%resultado = 2
fprintf('AUC CV máximo: %.4f\n', best_auc);%Imprime el AUC máximo con cuatro decimales. resultado = 1.0000 porque con 2 llegamos al 100% de separacion entre clases

tabla_cv = table(lvs_candidates(:), auc_cv(:), nze(:,1), ...
    'VariableNames', {'LVs','AUC_CV','NZ_beta'});

disp(tabla_cv); %resultado aqui
    % LVs    AUC_CV     NZ_beta  
    % ___    ______    __________
    % 
    %  1     0.9925    1.8432e+05
    %  2          1    1.8432e+05
    %  3          1    1.8432e+05
    %  4          1    1.8432e+05
    %  5          1    1.8432e+05
    %  6          1    1.8432e+05
    %  7          1    1.8432e+05
    %  8          1    1.8432e+05
    %  9          1    1.8432e+05
    % 10          1    1.8432e+05
writetable(tabla_cv, 'plsda_meda_validacion_cruzada.csv');

%% =========================================================
% 5) GRÁFICA DE VALIDACIÓN CRUZADA
% ==========================================================

figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

plot(ax, lvs_candidates, auc_cv, '-o', ...
     'LineWidth', 1.8, ...
     'MarkerSize', 7);

xlabel(ax, 'Número de variables latentes PLS', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'AUC en validación cruzada', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('PLS-DA MEDA: selección del número de variables latentes', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');

exportgraphics(gcf, 'plsda_meda_auc_cv.png', 'Resolution', 300);

%% =========================================================
% 6) VARIANZA CAPTURADA POR EL MODELO PLS
% ==========================================================
% varPls pertenece a la MEDA Toolbox y permite analizar la varianza
% capturada en X/Y en función del número de variables latentes.
% ==========================================================

if ~isempty(which('varPls'))

    [yvar, tvar] = varPls(X, Y, ...
        'LVs', 0:maxLV, ...
        'PreprocessingX', 2, ...
        'PreprocessingY', 0, ...
        'PlotScores', false);

    tabla_var = table((0:maxLV)', yvar(:), tvar(:), ...
        'VariableNames', {'LVs','Y_var_pct','Scores_var_pct'});

    writetable(tabla_var, 'plsda_meda_varianza_capturada.csv');

    figure('Color','w','Position',FIG_POS);

    ax = axes;
    ax.Position = AX_POS;

    plot(ax, 0:maxLV, yvar, '-o', ...
        'LineWidth', 1.8, ...
        'MarkerSize', 7);

    xlabel(ax, 'Número de variables latentes PLS', ...
           'FontSize', FS_EJES, ...
           'FontWeight', 'bold');

    ylabel(ax, 'Varianza capturada en Y (%)', ...
           'FontSize', FS_EJES, ...
           'FontWeight', 'bold');

    sgtitle('PLS-DA MEDA: varianza capturada en la variable de clase', ...
            'FontSize', FS_TITULO, ...
            'FontWeight', 'bold');

    set(ax, 'FontSize', FS_TICKS, ...
            'LineWidth', 1.2);

    grid(ax, 'on');

    exportgraphics(gcf, 'plsda_meda_varianza_y.png', 'Resolution', 300);
end

%% =========================================================
% 7) AJUSTAR MODELO FINAL PLS-DA CON SIMPLS
% ==========================================================
% Se usa simpls de la MEDA Toolbox.
%
% Xcs ya fue preprocesada con preprocess2D en el PCA.
% Y está codificada como +1/-1.
%
% El modelo resultante contiene:
%   model_plsda.loads
%   model_plsda.yloads
%   model_plsda.weights
%   model_plsda.altweights
%   model_plsda.scores
%   model_plsda.beta
% ==========================================================

lvs_model = 1:bestLV;%Crea el vector de variables latentes que se usarán en el modelo final. en este caso solo usamos 2

model_plsda = simpls(Xcs, Y, 'LVs', lvs_model);

model_plsda.av = average;%Añade al modelo la media usada para preprocesar X. viene del pca
model_plsda.sc = scale;%Añade al modelo la escala usada para preprocesar X. viene del pca tambien

disp('Campos disponibles en model_plsda:');
% Campos disponibles en model_plsda:
%     {'var'       }
%     {'lvs'       }
%     {'loads'     }
%     {'yloads'    }
%     {'weights'   }
%     {'altweights'}
%     {'scores'    }
%     {'beta'      }
%     {'type'      }
%     {'av'        }
%     {'sc'        }
disp(fieldnames(model_plsda));

T_pls = model_plsda.scores;%Extrae los scores del modelo PLS.Los scores son las coordenadas de las frases en el espacio latente PLS.
P_pls = model_plsda.loads;%Extrae los loadings de X. como pca
W_pls = model_plsda.weights;%Extrae los pesos del modelo PLS.Los pesos indican cómo se combinan las variables originales para construir las variables latentes.
B_pls = model_plsda.beta;%Extrae los coeficientes de regresión del modelo PLS.Estos coeficientes permiten predecir Y a partir de Xcs.

%% =========================================================
% 8) PREDICCIÓN DEL MODELO FINAL
% ==========================================================
% Como Y está codificada como +1/-1, el umbral natural es 0:
%
%   Y_pred >= 0 -> positive
%   Y_pred <  0 -> negative
% ==========================================================

Y_pred = Xcs * B_pls;%Calcula la salida continua del modelo PLS-DA.Como Y está codificada como +1/-1, la salida será un número continuo.

Y_class = ones(n_obs, 1);%Inicializa todas las predicciones como positivas.
Y_class(Y_pred < 0) = -1;%Las observaciones con salida menor que 0 se clasifican como negativas.

accuracy_train = mean(Y_class == Y);%Calcula el porcentaje de aciertos en entrenamiento.

fprintf('\nAccuracy en entrenamiento: %.2f %%\n', 100 * accuracy_train);%resultado = 100%

true_class = categorical(Y, [-1 1], {'Negativa','Positiva'});%Convierte la clase real a categorías legibles.
pred_class = categorical(Y_class, [-1 1], {'Negativa','Positiva'});%Convierte la clase predicha a categorías legibles.

df_pred = table((1:n_obs)', labels, Y, Y_pred, Y_class, ...
    'VariableNames', {'Observacion','Etiqueta','Y_real','Y_pred_continua','Y_pred_clase'});

writetable(df_pred, 'plsda_meda_predicciones.csv');

%% =========================================================
% 9) SCORE PLOT PLS-DA
% ==========================================================

if size(T_pls,2) >= 2 %Comprueba si hay al menos dos variables latentes.

    figure('Color','w','Position',FIG_POS);

    ax = axes;
    ax.Position = AX_POS;

    gscatter(ax, T_pls(:,1), T_pls(:,2), labels);

    xlabel(ax, 'LV1', ...
           'FontSize', FS_EJES, ...
           'FontWeight', 'bold');

    ylabel(ax, 'LV2', ...
           'FontSize', FS_EJES, ...
           'FontWeight', 'bold');

    sgtitle('PLS-DA MEDA: representación de las observaciones en LV1-LV2', ...
            'FontSize', FS_TITULO, ...
            'FontWeight', 'bold');

    set(ax, 'FontSize', FS_TICKS, ...
            'LineWidth', 1.2);

    leyenda = legend(ax, 'Location', 'best');
    leyenda.FontSize = FS_LEYENDA;
    leyenda.FontWeight = 'bold';

    grid(ax, 'on');

    exportgraphics(gcf, 'plsda_meda_scores_lv1_lv2.png', 'Resolution', 300);

else
% Si solo hay una variable latente, no se puede hacer LV1-LV2.
% parte omitible en este caso ya que tenemos las dos componentes
% Entonces se representa LV1 frente al índice de observación.
    figure('Color','w','Position',FIG_POS);

    ax = axes;
    ax.Position = AX_POS;

    gscatter(ax, (1:n_obs)', T_pls(:,1), labels);

    xlabel(ax, 'Índice de observación', ...
           'FontSize', FS_EJES, ...
           'FontWeight', 'bold');

    ylabel(ax, 'LV1', ...
           'FontSize', FS_EJES, ...
           'FontWeight', 'bold');

    sgtitle('PLS-DA MEDA: scores en LV1', ...
            'FontSize', FS_TITULO, ...
            'FontWeight', 'bold');

    set(ax, 'FontSize', FS_TICKS, ...
            'LineWidth', 1.2);

    grid(ax, 'on');

    exportgraphics(gcf, 'plsda_meda_scores_lv1.png', 'Resolution', 300);
end

%% =========================================================
% 10) MATRIZ DE CONFUSIÓN
% ==========================================================

figure('Color','w','Position',FIG_POS);

cm = confusionchart(true_class, pred_class, ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

cm.FontSize = FS_TICKS;
cm.Title = '';

sgtitle('PLS-DA MEDA: matriz de confusión', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

exportgraphics(gcf, 'plsda_meda_matriz_confusion.png', 'Resolution', 300);

%% =========================================================
% 11) PREDICCIÓN CONTINUA DEL MODELO
% ==========================================================

figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

hold(ax, 'on');

idx_pos = find(Y == 1);
idx_neg = find(Y == -1);

scatter(ax, idx_pos, Y_pred(idx_pos), 70, 'filled');
scatter(ax, idx_neg, Y_pred(idx_neg), 70, 'filled');

hthr = yline(ax, 0, '--k', 'Umbral 0');
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

sgtitle('PLS-DA MEDA: predicción continua del modelo', ...
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

exportgraphics(gcf, 'plsda_meda_prediccion_continua.png', 'Resolution', 300);

%% =========================================================
% 12) IMPORTANCIA DE VARIABLES MEDIANTE COEFICIENTES PLS
% ==========================================================
% La MEDA Toolbox devuelve model_plsda.beta.
% Cada coeficiente se asocia a una variable original de atención.
%
% Como medida de importancia se usa |beta|.
% El signo indica dirección:
%   beta > 0 -> contribuye hacia la clase positiva
%   beta < 0 -> contribuye hacia la clase negativa
% ==========================================================

coef_pls = B_pls(:);%Convierte los coeficientes PLS en vector columna.Cada elemento corresponde a una variable de atención.
importance_pls = abs(coef_pls);%Calcula la importancia como valor absoluto del coeficiente.

[importance_sorted, idx_sorted_all] = sort(importance_pls, 'descend');%Ordena las variables de mayor a menor importancia.

K = 50;%igual que oMEDA
topK_idx = idx_sorted_all(1:K);
topK_vals = coef_pls(topK_idx);
topK_abs  = importance_pls(topK_idx);

tabla_topK = table(topK_idx, topK_vals, topK_abs, ...
    'VariableNames', {'Variable','Coeficiente_PLS','Abs_Coeficiente_PLS'});

fprintf('\nTop %d variables según |coeficiente PLS-DA|:\n', K);
disp(tabla_topK);
% Top 50 variables según |coeficiente PLS-DA|:
%     Variable    Coeficiente_PLS    Abs_Coeficiente_PLS
%     ________    _______________    ___________________
% 
%       8383         0.0019111            0.0019111     
%       8384         0.0019106            0.0019106     
%       8001        -0.0018957            0.0018957     
%       7809        -0.0018807            0.0018807     
%       8433        -0.0018764            0.0018764     
%       8705         0.0018599            0.0018599     
%       8193        -0.0018453            0.0018453     
%       8881         0.0018309            0.0018309     
%       8665        -0.0018227            0.0018227     
%       8889         0.0018175            0.0018175     
%       8441        -0.0017858            0.0017858     
%       7817        -0.0017827            0.0017827     
%       8513         0.0017817            0.0017817     
%       8007         0.0017789            0.0017789     
%       9156        -0.0017747            0.0017747     
%       9025        -0.0017739            0.0017739     
%       8524         0.0017697            0.0017697     
%       9040        -0.0017692            0.0017692     
%       7815          0.001759             0.001759     
%       9039        -0.0017526            0.0017526     
%       9160         0.0017502            0.0017502     
%       8519        -0.0017479            0.0017479     
%       8263         0.0017457            0.0017457     
%       8520        -0.0017397            0.0017397     
%       9159         0.0017279            0.0017279     
%       8376         0.0017163            0.0017163     
%       8200         0.0017122            0.0017122     
%       8375          0.001709             0.001709     
%       7929          0.001708             0.001708     
%       9048        -0.0017058            0.0017058     
%       8692        -0.0017034            0.0017034     
%       7752          0.001703             0.001703     
%       8700        -0.0016988            0.0016988     
%       8530         0.0016934            0.0016934     
%       8369        -0.0016855            0.0016855     
%       7879        -0.0016823            0.0016823     
%       8199         0.0016776            0.0016776     
%       8442        -0.0016714            0.0016714     
%       7751         0.0016699            0.0016699     
%       8521         0.0016693            0.0016693     
%       8769         0.0016672            0.0016672     
%       8447          0.001663             0.001663     
%       7873          0.001659             0.001659     
%       9047        -0.0016528            0.0016528     
%       8649        -0.0016486            0.0016486     
%       9056         -0.001637             0.001637     
%       8516         0.0016347            0.0016347     
%       7816         0.0016315            0.0016315     
%       8625          0.001627             0.001627     
%       9064        -0.0016268            0.0016268 
writetable(tabla_topK, 'plsda_meda_top50_variables_coeficientes.csv');

%% =========================================================
% 13) HISTOGRAMA DE IMPORTANCIA POR COEFICIENTES
% ==========================================================

figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

histogram(ax, importance_pls, 50);

xlabel(ax, '|Coeficiente PLS-DA|', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Frecuencia', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('PLS-DA MEDA: distribución de importancia de variables', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');

exportgraphics(gcf, 'plsda_meda_hist_importancia_coeficientes.png', 'Resolution', 300);

%% =========================================================
% 14) AGRUPAR IMPORTANCIA POR CAPA Y HEAD
% ==========================================================
% 9216 = 12 capas * 12 heads * 8 tokens origen * 8 tokens destino
% ==========================================================
%igual que omeda
n_layers = 12;
n_heads  = 12;
n_tokens = 8;

if n_vars ~= n_layers*n_heads*n_tokens*n_tokens
    error('El número de variables no coincide con 12*12*8*8 = 9216.');
end

imp_by_layer_pls = zeros(n_layers, 1);
imp_by_head_pls  = zeros(n_layers, n_heads);

for var_idx = 1:n_vars

    zero_idx = var_idx - 1;

    layer_idx = floor(zero_idx / (n_heads*n_tokens*n_tokens));
    rem1      = mod(zero_idx, n_heads*n_tokens*n_tokens);

    head_idx  = floor(rem1 / (n_tokens*n_tokens));

    imp_by_layer_pls(layer_idx + 1) = imp_by_layer_pls(layer_idx + 1) + importance_pls(var_idx);
    imp_by_head_pls(layer_idx + 1, head_idx + 1) = imp_by_head_pls(layer_idx + 1, head_idx + 1) + importance_pls(var_idx);
end

tabla_layer_pls = table((1:n_layers)', imp_by_layer_pls, ...
    100*imp_by_layer_pls/sum(imp_by_layer_pls), ...
    'VariableNames', {'Layer','Importancia','Importancia_pct'});

writetable(tabla_layer_pls, 'plsda_meda_importancia_layer.csv');

%% =========================================================
% 15) IMPORTANCIA POR CAPA
% ==========================================================
%igual que omeda
figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = AX_POS;

bar(ax, 1:n_layers, imp_by_layer_pls);

xlabel(ax, 'Layer', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Importancia acumulada |coeficiente PLS-DA|', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('PLS-DA MEDA: importancia por capa', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

xticks(ax, 1:n_layers);
grid(ax, 'on');

exportgraphics(gcf, 'plsda_meda_importancia_por_capa.png', 'Resolution', 300);

%% =========================================================
% 16) HEATMAP LAYER-HEAD
% ==========================================================
%igual que omeda
figure('Color','w','Position',FIG_POS);

ax = axes;
ax.Position = [0.10 0.14 0.76 0.72];

imagesc(ax, 1:n_heads, 1:n_layers, imp_by_head_pls);

cb = colorbar(ax);
cb.FontSize = FS_TICKS;
cb.Label.String = 'Importancia acumulada |coeficiente PLS-DA|';
cb.Label.FontSize = FS_EJES;
cb.Label.FontWeight = 'bold';

xlabel(ax, 'Head', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Layer', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('PLS-DA MEDA: importancia por layer-head', ...
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

exportgraphics(gcf, 'plsda_meda_importancia_layer_head.png', 'Resolution', 300);

%% =========================================================
% 17) IMPORTANCIA POR TOKEN EN PLS-DA
% ==========================================================
% Se acumula |coeficiente PLS-DA| por:
%   - token origen
%   - token destino
%   - combinación origen + destino
% ==========================================================
%igual que omeda solo que en vez de los valores de omeda con los
%coeficientes de plsda
token_labels = ["[CLS]"; "Token 2"; "Token 3"; "Token 4"; ...
                "Token 5"; "Token 6"; "."; "[SEP]"];

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

    imp_tokenO_plsda(tokenO_idx) = imp_tokenO_plsda(tokenO_idx) + importance_pls(var_idx);
    imp_tokenD_plsda(tokenD_idx) = imp_tokenD_plsda(tokenD_idx) + importance_pls(var_idx);
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

writetable(tabla_plsda_tokens, 'plsda_meda_importancia_tokens.csv');

figure('Color','w','Position',FIG_POS);

bar([imp_tokenO_plsda_pct, imp_tokenD_plsda_pct, imp_token_comb_plsda_pct]);

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

title('PLS-DA MEDA: importancia relativa por token', ...
      'FontSize', FS_TITULO, ...
      'FontWeight', 'bold');

grid on;

set(gca, 'FontSize', FS_TICKS, ...
         'LineWidth', 1.2);

exportgraphics(gcf, 'plsda_meda_importancia_tokens.png', 'Resolution', 300);

%% =========================================================
% 18) COMPARACIÓN CON oMEDA MEDA, SI EXISTE
% ==========================================================
% Este bloque compara la importancia por capas de PLS-DA con el
% oMEDA calculado previamente con la MEDA Toolbox.
% ==========================================================

if exist('omeda_attention_results_meda.mat', 'file')

    S_omeda = load('omeda_attention_results_meda.mat');

    if isfield(S_omeda, 'importance')

        importance_omeda = S_omeda.importance(:);

    elseif isfield(S_omeda, 'omeda_vec')

        importance_omeda = abs(S_omeda.omeda_vec(:));

    else
        importance_omeda = [];
    end

    if ~isempty(importance_omeda) && numel(importance_omeda) == n_vars

        imp_by_layer_omeda = zeros(n_layers, 1);

        for var_idx = 1:n_vars

            zero_idx = var_idx - 1;
            layer_idx = floor(zero_idx / (n_heads*n_tokens*n_tokens));

            imp_by_layer_omeda(layer_idx + 1) = imp_by_layer_omeda(layer_idx + 1) + importance_omeda(var_idx);
        end

        imp_by_layer_omeda_norm = imp_by_layer_omeda / sum(imp_by_layer_omeda);
        imp_by_layer_pls_norm   = imp_by_layer_pls   / sum(imp_by_layer_pls);

        figure('Color','w','Position',FIG_POS);

        ax = axes;
        ax.Position = AX_POS;

        bar(ax, 1:n_layers, [imp_by_layer_omeda_norm, imp_by_layer_pls_norm]);

        xlabel(ax, 'Layer', ...
               'FontSize', FS_EJES, ...
               'FontWeight', 'bold');

        ylabel(ax, 'Importancia normalizada', ...
               'FontSize', FS_EJES, ...
               'FontWeight', 'bold');

        sgtitle('Comparación por capas: oMEDA MEDA vs PLS-DA MEDA', ...
                'FontSize', FS_TITULO, ...
                'FontWeight', 'bold');

        leyenda = legend(ax, 'oMEDA', 'PLS-DA', ...
                         'Location', 'best');
        leyenda.FontSize = FS_LEYENDA;
        leyenda.FontWeight = 'bold';

        set(ax, 'FontSize', FS_TICKS, ...
                'LineWidth', 1.2);

        xticks(ax, 1:n_layers);
        grid(ax, 'on');

        exportgraphics(gcf, 'comparacion_omeda_meda_plsda_meda_por_capa.png', 'Resolution', 300);

        corr_var = corr(importance_omeda(:), importance_pls(:), 'Type', 'Spearman');

        fprintf('\nCorrelación Spearman entre |oMEDA| y |coeficientes PLS-DA|: %.4f\n', corr_var);

        topK_omeda = idx_topK_local(importance_omeda, K);
        topK_pls   = idx_topK_local(importance_pls, K);

        overlap_topK = numel(intersect(topK_omeda, topK_pls));

        fprintf('Solapamiento Top-%d entre oMEDA y PLS-DA: %d variables\n', K, overlap_topK);

        figure('Color','w','Position',FIG_POS);

        ax = axes;
        ax.Position = AX_POS;

        scatter(ax, importance_omeda, importance_pls, 20, 'filled');

        xlabel(ax, '|oMEDA|', ...
               'FontSize', FS_EJES, ...
               'FontWeight', 'bold');

        ylabel(ax, '|Coeficiente PLS-DA|', ...
               'FontSize', FS_EJES, ...
               'FontWeight', 'bold');

        sgtitle('Comparación variable a variable: oMEDA MEDA vs PLS-DA MEDA', ...
                'FontSize', FS_TITULO, ...
                'FontWeight', 'bold');

        set(ax, 'FontSize', FS_TICKS, ...
                'LineWidth', 1.2);

        grid(ax, 'on');

        exportgraphics(gcf, 'scatter_omeda_meda_vs_plsda_meda.png', 'Resolution', 300);

    else
        fprintf('\nNo se encontró un vector oMEDA compatible. Se omite comparación directa.\n');
    end

else
    fprintf('\nNo se encontró omeda_attention_results_meda.mat. Se omite comparación directa con oMEDA.\n');
end

%% =========================================================
% 19) GUARDAR RESULTADOS PRINCIPALES
% ==========================================================

save('plsda_results_meda.mat', ...
    'Y', ...
    'Y_pred', ...
    'Y_class', ...
    'true_class', ...
    'pred_class', ...
    'bestLV', ...
    'best_auc', ...
    'auc_cv', ...
    'nze', ...
    'model_plsda', ...
    'T_pls', ...
    'P_pls', ...
    'W_pls', ...
    'B_pls', ...
    'coef_pls', ...
    'importance_pls', ...
    'importance_sorted', ...
    'idx_sorted_all', ...
    'imp_by_layer_pls', ...
    'imp_by_head_pls', ...
    'imp_tokenO_plsda', ...
    'imp_tokenD_plsda', ...
    'imp_token_comb_plsda', ...
    'imp_tokenO_plsda_pct', ...
    'imp_tokenD_plsda_pct', ...
    'imp_token_comb_plsda_pct', ...
    'labels');

fprintf('\nAnálisis PLS-DA con MEDA Toolbox finalizado.\n');
fprintf('Resultados guardados en plsda_results_meda.mat\n');

%% =========================================================
% FUNCIONES AUXILIARES LOCALES
% ==========================================================
% Esta función solo ordena índices. No calcula el modelo PLS-DA.
% El modelo PLS-DA se ha calculado únicamente con funciones de la
% MEDA Toolbox.
% ==========================================================

function idx_top = idx_topK_local(values, K)
    [~, idx_sorted] = sort(values, 'descend');
    idx_top = idx_sorted(1:K);
end