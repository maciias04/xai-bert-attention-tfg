%% =========================================================
%  ANALISIS INDIVIDUALIZADO DE TOKENS CON oMEDA, PLS-DA Y ASCA
%  usando exclusivamente funciones de la MEDA Toolbox para el modelado
%
%  Autor: Adrian Macias Caballero
%
%  Objetivo:
%  - Analizar individualmente las 40 frases del experimento.
%  - Obtener, para cada frase, que posiciones de token contribuyen mas
%    segun oMEDA-PCA, oMEDA-PLS y ASCA.
%  - Agrupar las 9216 variables de atencion por token origen y destino.
%  - Generar rankings individuales y rankings globales por palabra/token.
%
%  Estructura de variables:
%  9216 = 12 layers x 12 heads x 8 tokens origen x 8 tokens destino
% ==========================================================

clear;
clc;
close all;

%% =========================================================
% 0) CONFIGURACION DEL USUARIO
% ==========================================================
% Cambia esta ruta por la ruta real de tu MEDA Toolbox.
% La carpeta debe contener funciones como omeda, pcaEig, simpls,
% parglm, asca, preprocess2D y preprocess2Dapp.
% ==========================================================

toolbox_path = 'C:\Users\Usuario\Desktop\UNI\Cuarto de Carrera Teleco\doctoradopepe\MEDA Shared Materials-20260301T114916Z-3-001\MEDA Shared Materials\MEDA-Toolbox-master\MEDA-Toolbox-master\toolbox';

if exist(toolbox_path, 'dir')
    addpath(genpath(toolbox_path));
else
    warning('No se ha encontrado toolbox_path. Si la toolbox ya esta en el path, puedes ignorar este aviso.');
end

DATA_FILE = 'pca_attention_results_meda.mat';

% Numero de tokens que se guardan por frase y metodo.
TOP_K = 8;

% Si es true, genera 120 figuras individuales: 40 frases x 3 metodos.
% Si solo quieres tablas CSV, dejalo en false.
PLOT_INDIVIDUAL = true;

% Preprocesamiento usado en los modelos de la toolbox.
% 0 = sin preprocesar, 1 = centrado, 2 = autoscalado.
PREP_X = 2;
PREP_Y_PLS = 0;

% Numero maximo de LVs a probar si no existe plsda_results_meda.mat.
MAX_LV_CV = 10;

% Numero de permutaciones en ASCA.
N_PERM_ASCA = 1000;

% Carpetas de salida.
OUT_DIR = 'resultados_individualizados_tokens';
FIG_DIR = fullfile(OUT_DIR, 'figuras');
INDIV_FIG_DIR = fullfile(FIG_DIR, 'individuales');

if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end
if ~exist(FIG_DIR, 'dir'), mkdir(FIG_DIR); end
if PLOT_INDIVIDUAL && ~exist(INDIV_FIG_DIR, 'dir'), mkdir(INDIV_FIG_DIR); end

%% =========================================================
% 1) COMPROBAR FUNCIONES DE LA MEDA TOOLBOX
% ==========================================================

required_functions = {'preprocess2D','preprocess2Dapp','pcaEig','omeda', ...
                      'simpls','crossvalPlsDA','parglm','asca'};

for k = 1:numel(required_functions)
    if isempty(which(required_functions{k}))
        error('No se encuentra la funcion %s. Revisa que la MEDA Toolbox este en el path.', required_functions{k});
    end
end

disp('MEDA Toolbox encontrada correctamente.');

%% =========================================================
% 2) CARGAR DATOS
% ==========================================================

if ~exist(DATA_FILE, 'file')
    error('No se encuentra %s. Ejecuta antes el script de extraccion/PCA.', DATA_FILE);
end

load(DATA_FILE);

if ~exist('X', 'var')
    error('El archivo %s no contiene la matriz X.', DATA_FILE);
end

if ~exist('sentences', 'var')
    error('El archivo %s no contiene la variable sentences.', DATA_FILE);
end

if ~exist('true_label', 'var')
    error('El archivo %s no contiene la variable true_label.', DATA_FILE);
end

X = double(X);
sentences = string(sentences(:));%Convierte las frases a tipo string y las pone como vector columna.
labels = lower(string(true_label(:)));

[n_obs, n_vars] = size(X);%tamaño de x = 40 x 9216

fprintf('Observaciones: %d\n', n_obs);
fprintf('Variables: %d\n', n_vars);

%% =========================================================
% 3) CODIFICAR CLASES Y TOKENS DE CADA FRASE
% ==========================================================
% Y se usa para PLS-DA:
%   positive -> +1
%   negative -> -1
%
% class_id se usa para ASCA:
%   positive -> 1
%   negative -> 2
% ==========================================================

idx_pos = labels == "positive" | labels == "positiva" | labels == "positivas"; %vale true para frases positivas.
idx_neg = labels == "negative" | labels == "negativa" | labels == "negativas"; %vale true para frases negativas.

Y = zeros(n_obs, 1);
Y(idx_pos) =  1;
Y(idx_neg) = -1;

if any(Y == 0)
    error('Hay etiquetas que no se han podido codificar como positive/negative.');
end

class_id = zeros(n_obs, 1);%Crea la codificación para ASCA.
class_id(idx_pos) = 1;%diseño experimental de ASCA.
class_id(idx_neg) = 2;

% Parejas de frases: positiva i con negativa i.
idx_pos_order = find(idx_pos);
idx_neg_order = find(idx_neg);
n_pair = min(numel(idx_pos_order), numel(idx_neg_order));%20 pares de frases en total

pair_id = zeros(n_obs, 1);%asigna el mismo numero de pareja a cada frase positiva y negativa, la pos 1 con la neg 1 es la pareja 1
pair_id(idx_pos_order(1:n_pair)) = (1:n_pair)';
pair_id(idx_neg_order(1:n_pair)) = (1:n_pair)';

idx_unpaired = find(pair_id == 0);
if ~isempty(idx_unpaired)
    pair_id(idx_unpaired) = n_pair + (1:numel(idx_unpaired))';
end

% Tokenizacion simple coherente con el experimento:
% [CLS] + 5 palabras + . + [SEP]
token_text = strings(n_obs, 8); %[CLS] palabra1 palabra2 palabra3 palabra4 palabra5 . [SEP] 40x8

for i = 1:n_obs  %convierte las 40 frases a los tokens de BERT bien para analizarlas
    s = lower(strtrim(sentences(i)));
    s = replace(s, ".", " .");
    parts = split(s);
    parts(parts == "") = [];

    if numel(parts) == 6
        token_text(i,:) = ["[CLS]", reshape(parts, 1, []), "[SEP]"];
    else
        warning('La frase %d no tiene 5 palabras + punto. Se usaran etiquetas genericas.', i);
        token_text(i,:) = ["[CLS]", "Token 2", "Token 3", "Token 4", ...
                           "Token 5", "Token 6", ".", "[SEP]"];
    end
end

%% =========================================================
% 4) MAPEO DE LAS 9216 VARIABLES A LAYER, HEAD, TOKEN ORIGEN Y DESTINO
% ==========================================================

n_layers = 12;
n_heads  = 12;
n_tokens = 8;

if n_vars ~= n_layers*n_heads*n_tokens*n_tokens
    error('El numero de variables no coincide con 12*12*8*8 = 9216.');
end

var_index = (1:n_vars)';
zero_idx = var_index - 1;

layer_idx = floor(zero_idx / (n_heads*n_tokens*n_tokens)) + 1;%Calcula a qué capa pertenece cada variable.
rem1 = mod(zero_idx, n_heads*n_tokens*n_tokens);

head_idx = floor(rem1 / (n_tokens*n_tokens)) + 1;%Calcula la cabeza de atención.
rem2 = mod(rem1, n_tokens*n_tokens);

tokenO_idx = floor(rem2 / n_tokens) + 1;%Calcula el token origen.
tokenD_idx = mod(rem2, n_tokens) + 1;%Calcula el token destino.

%% =========================================================
% 5) MODELO oMEDA-PCA INDIVIDUALIZADO
% ==========================================================
% Se calcula un modelo PCA con funciones de la MEDA Toolbox y despues
% se obtiene un vector oMEDA para cada frase individual.
% ==========================================================

fprintf('\nCalculando vectores individuales oMEDA-PCA...\n');

[Xcs_pca, mean_pca, scale_pca] = preprocess2D(X, 'Preprocessing', PREP_X);

if exist('A', 'var')
    n_pcs_omeda = min([A, rank(Xcs_pca), n_obs-1]);
else
    n_pcs_omeda = min([rank(Xcs_pca), n_obs-1]);
end

pcs_omeda = 1:n_pcs_omeda;%Define las componentes PCA que se usarán. 29 que es el valor de A
model_pca_ind = pcaEig(Xcs_pca, 'PCs', pcs_omeda);%Calcula el modelo PCA
P_pca_ind = model_pca_ind.loads;%Extrae los loadings del PCA.

V_omeda = zeros(n_obs, n_vars);%Crea una matriz para guardar los vectores individuales oMEDA.

for i = 1:n_obs
    testcs = preprocess2Dapp(X(i,:), mean_pca, 'Scale', scale_pca);%frase preprocesada
    vec_i = omeda(testcs, 1, P_pca_ind);%Calcula el vector oMEDA individual de esa frase. Este vector tiene 9216 variables.
    V_omeda(i,:) = vec_i(:)';
end

%% =========================================================
% 6) MODELO PLS-DA INDIVIDUALIZADO MEDIANTE oMEDA-PLS
% ==========================================================
% Se entrena PLS-DA con simpls y se obtiene un vector oMEDA-PLS
% para cada frase individual.
% ==========================================================

fprintf('Calculando vectores individuales PLS-DA / oMEDA-PLS...\n');

if exist('plsda_results_meda.mat', 'file')
    S_pls = load('plsda_results_meda.mat', 'bestLV');
    if isfield(S_pls, 'bestLV')
        bestLV = S_pls.bestLV;%el número óptimo de variables latentes seleccionado antes.
    else
        bestLV = 1;
    end
else%sino existe el bestLV lo calcula aplicando el modelo PLS-DA que aplicamos antes
    maxLV = min([MAX_LV_CV, rank(X), n_obs-1]);
    lvs_candidates = 1:maxLV;
    maxBlock = min(sum(Y == 1), sum(Y == -1));

    [AUC_raw, ~] = crossvalPlsDA(X, Y, ...
        'LVs', lvs_candidates, ...
        'VarNumber', n_vars, ...
        'MaxBlock', maxBlock, ...
        'PreprocessingX', PREP_X, ...
        'PreprocessingY', PREP_Y_PLS, ...
        'Selection', 'SR', ...
        'Plot', false);

    if ndims(AUC_raw) == 3
        auc_cv = squeeze(AUC_raw(:,1,1));
    else
        auc_cv = AUC_raw(:,1);
    end

    auc_cv_clean = auc_cv;
    auc_cv_clean(isnan(auc_cv_clean)) = -Inf;
    [~, idx_best] = max(auc_cv_clean);
    bestLV = lvs_candidates(idx_best);
end
%esta parte del bucle es skipeable porque ya lo habiamos hecho en nuestro
%anterior modelo
bestLV = max(1, bestLV);
lvs_pls = 1:bestLV;

[Xcs_pls, mean_pls, scale_pls] = preprocess2D(X, 'Preprocessing', PREP_X);
Ycs = preprocess2D(Y, 'Preprocessing', PREP_Y_PLS);

model_pls_ind = simpls(Xcs_pls, Ycs, 'LVs', lvs_pls);
R_pls = model_pls_ind.altweights;%son los pesos alternativos.(w)
P_pls = model_pls_ind.loads;%son los loadings.

V_plsda = zeros(n_obs, n_vars);%Crea una matriz para guardar los vectores individuales PLS-DA.

for i = 1:n_obs
    testcs = preprocess2Dapp(X(i,:), mean_pls, 'Scale', scale_pls);%Preprocesa la frase usando la media y escala de PLS.
    vec_i = omeda(testcs, 1, R_pls, 'OutSubspace', P_pls);%Calcula un vector de importancia tipo oMEDA, pero usando el subespacio PLS.
    V_plsda(i,:) = vec_i(:)';
end

%% =========================================================
% 7) MODELO ASCA INDIVIDUALIZADO
% ==========================================================
% Se sigue la idea propuesta por el tutor:
%   F = [clase, pareja]
%   model = asca(parglm(X,F))
%   X2 = X - residuals
%   vecF = omedaPca(X2, [], X2(i,:), 1)
%
% Para evitar que omedaPca genere una figura por cada frase, aqui se
% reproduce el calculo con funciones internas de la toolbox:
% preprocess2D + pcaEig + omeda.
% ==========================================================

fprintf('Calculando vectores individuales ASCA...\n');

F = [class_id, pair_id];%Construye la matriz de diseño experimental para ASCA. columna 1 clase positiva/negativa, columna 2 pareja de frases.

[tabla_anova_asca, parglmo] = parglm(X, F, ...
    'Model', 'linear', ...
    'Preprocessing', PREP_X, ...
    'Permutations', N_PERM_ASCA, ...
    'Ts', 2);

ascao = asca(parglmo);

% Matriz ajustada por ASCA. Se usa ascao.data porque parglm trabaja en
% escala preprocesada internamente. Esto evita mezclar X cruda con residuals
% escalados.
X_asca_fit = ascao.data - ascao.residuals;
%ascao.data son los datos usados internamente por ASCA, normalmente ya preprocesados.
%ascao.residuals es la parte residual no explicada por el modelo ASCA.
%Esto sigue la idea de tu tutor:X2 = X - residuals
[Xcs_asca, mean_asca, scale_asca] = preprocess2D(X_asca_fit, 'Preprocessing', PREP_X);
n_pcs_asca = min([rank(Xcs_asca), n_obs-1]);%Calcula cuántas componentes PCA usar para la parte individualizada de ASCA.
pcs_asca = 1:n_pcs_asca;
model_asca_ind = pcaEig(Xcs_asca, 'PCs', pcs_asca);
P_asca_ind = model_asca_ind.loads;

V_asca = zeros(n_obs, n_vars);%Crea la matriz donde se guardarán los vectores individuales ASCA.

for i = 1:n_obs
    testcs = preprocess2Dapp(X_asca_fit(i,:), mean_asca, 'Scale', scale_asca);%Preprocesa la frase i dentro del espacio ASCA ajustado.
    vec_i = omeda(testcs, 1, P_asca_ind);%Calcula el vector individual de importancia sobre el subespacio ASCA.
    V_asca(i,:) = vec_i(:)';
end

%% =========================================================
% 8) ACUMULAR IMPORTANCIA POR TOKEN PARA CADA FRASE Y METODO
% ==========================================================
% Para cada vector individual de 9216 variables:
%   - se toma valor absoluto
%   - se suma por token origen
%   - se suma por token destino
%   - se calcula una medida combinada origen + destino
% ==========================================================

methods = ["oMEDA-PCA", "PLS-DA", "ASCA"];
V_all = cat(3, V_omeda, V_plsda, V_asca);

imp_origin = zeros(n_obs, n_tokens, numel(methods));
imp_dest   = zeros(n_obs, n_tokens, numel(methods));
imp_comb   = zeros(n_obs, n_tokens, numel(methods));

imp_origin_pct = zeros(n_obs, n_tokens, numel(methods));
imp_dest_pct   = zeros(n_obs, n_tokens, numel(methods));
imp_comb_pct   = zeros(n_obs, n_tokens, numel(methods));

for m = 1:numel(methods)
    V_method = V_all(:,:,m);

    for i = 1:n_obs
        v_abs = abs(V_method(i,:))';%Toma el valor absoluto del vector individual de 9216 variables.

        io = accumarray(tokenO_idx, v_abs, [n_tokens 1], @sum, 0);%Suma las importancias por token origen.
        id = accumarray(tokenD_idx, v_abs, [n_tokens 1], @sum, 0);%Suma las importancias por token destino.
        ic = io + id;%Calcula la importancia combinada
%Guarda las importancias absolutas.
        imp_origin(i,:,m) = io';
        imp_dest(i,:,m)   = id';
        imp_comb(i,:,m)   = ic';
%Normaliza la importancia para que sume 100 %.
        if sum(io) > 0
            imp_origin_pct(i,:,m) = 100 * io' / sum(io);
        end
        if sum(id) > 0
            imp_dest_pct(i,:,m) = 100 * id' / sum(id);
        end
        if sum(ic) > 0
            imp_comb_pct(i,:,m) = 100 * ic' / sum(ic);
        end
    end
end

%% =========================================================
% 9) TABLA COMPLETA: IMPORTANCIA POR TOKEN, FRASE Y METODO
% ==========================================================

n_rows = numel(methods) * n_obs * n_tokens;%3×40×8=960
%Después crea columnas vacías:
Method = strings(n_rows,1);%metodo
Observation = zeros(n_rows,1);%numero de frase
Sentence = strings(n_rows,1);%frase
TrueLabel = strings(n_rows,1);%etiqueta real
TokenPosition = zeros(n_rows,1);%posicion del token
Token = strings(n_rows,1);%palabra real
ImportanceOrigin = zeros(n_rows,1);%importancia cuando es token origen
ImportanceDestination = zeros(n_rows,1);%importancia cuando es token destino
ImportanceCombined = zeros(n_rows,1);%importancia combinada
ImportanceOriginPct = zeros(n_rows,1);%mismas importancias normalizadas a porcentaje para que esten en la misma escala
ImportanceDestinationPct = zeros(n_rows,1);
ImportanceCombinedPct = zeros(n_rows,1);

r = 0;%contador de filas
for m = 1:numel(methods)%primer bucle de metodos, primero omeda, segundo pls y tercero asca
    for i = 1:n_obs%recorrer las 40 frases en los 3 metodos
        for t = 1:n_tokens%recorrer los 8 tokens de las 40 frases de los 3 metodos
            r = r + 1;
            Method(r) = methods(m);
            Observation(r) = i;
            Sentence(r) = sentences(i);
            TrueLabel(r) = labels(i);
            TokenPosition(r) = t;
            Token(r) = token_text(i,t);
            ImportanceOrigin(r) = imp_origin(i,t,m);
            ImportanceDestination(r) = imp_dest(i,t,m);
            ImportanceCombined(r) = imp_comb(i,t,m);
            ImportanceOriginPct(r) = imp_origin_pct(i,t,m);
            ImportanceDestinationPct(r) = imp_dest_pct(i,t,m);
            ImportanceCombinedPct(r) = imp_comb_pct(i,t,m);
        end
    end
end

T_all_tokens = table(Method, Observation, Sentence, TrueLabel, TokenPosition, Token, ...
    ImportanceOrigin, ImportanceDestination, ImportanceCombined, ...
    ImportanceOriginPct, ImportanceDestinationPct, ImportanceCombinedPct);

writetable(T_all_tokens, fullfile(OUT_DIR, 'importancia_tokens_individual_omeda_plsda_asca.csv'));

%% =========================================================
% 10) TABLA DE TOP TOKENS POR FRASE Y METODO
% ==========================================================
% Se guardan dos rankings:
%   - all_tokens: incluye [CLS], punto y [SEP]
%   - real_tokens: excluye [CLS], punto y [SEP]
% ==========================================================

n_rows_top = numel(methods) * n_obs * TOP_K * 2;%Calcula el tamaño máximo de la tabla de rankings.
%Multiplica por 2 porque guarda dos tipos de ranking, el de todos los
%tokens y el de los tokens reales(es decir, palabras)
Top_Method = strings(n_rows_top,1);
Top_Observation = zeros(n_rows_top,1);
Top_Sentence = strings(n_rows_top,1);
Top_TrueLabel = strings(n_rows_top,1);
Top_RankingType = strings(n_rows_top,1);
Top_Rank = zeros(n_rows_top,1);
Top_TokenPosition = zeros(n_rows_top,1);
Top_Token = strings(n_rows_top,1);
Top_ImportanceCombined = zeros(n_rows_top,1);
Top_ImportanceCombinedPct = zeros(n_rows_top,1);
Top_ImportanceOriginPct = zeros(n_rows_top,1);
Top_ImportanceDestinationPct = zeros(n_rows_top,1);

r = 0;
for m = 1:numel(methods)
    for i = 1:n_obs
        values = squeeze(imp_comb_pct(i,:,m));%Extrae la importancia combinada porcentual de los 8 tokens para esa frase y método.

        % Ranking con todos los tokens.
        [~, ord_all] = sort(values, 'descend');%Ordena los tokens de mayor a menor importancia.
        ord_all = ord_all(1:min(TOP_K, numel(ord_all)));%Se queda con los TOP_K primeros.en este caso coge los 8 como le he dicho para ver todos

        for k = 1:numel(ord_all)
            t = ord_all(k);
            r = r + 1;
            Top_Method(r) = methods(m);
            Top_Observation(r) = i;
            Top_Sentence(r) = sentences(i);
            Top_TrueLabel(r) = labels(i);
            Top_RankingType(r) = "all_tokens";
            Top_Rank(r) = k;
            Top_TokenPosition(r) = t;
            Top_Token(r) = token_text(i,t);
            Top_ImportanceCombined(r) = imp_comb(i,t,m);
            Top_ImportanceCombinedPct(r) = imp_comb_pct(i,t,m);
            Top_ImportanceOriginPct(r) = imp_origin_pct(i,t,m);
            Top_ImportanceDestinationPct(r) = imp_dest_pct(i,t,m);
        end

        % Ranking solo de palabras reales. maskea todo lo que no sean
        % palabras
        real_mask = ~(token_text(i,:) == "[CLS]" | token_text(i,:) == "[SEP]" | token_text(i,:) == ".");
        real_positions = find(real_mask);
        real_values = values(real_positions);
        [~, ord_real_local] = sort(real_values, 'descend');
        ord_real_local = ord_real_local(1:min(TOP_K, numel(ord_real_local)));
        ord_real = real_positions(ord_real_local);

        for k = 1:numel(ord_real)
            t = ord_real(k);
            r = r + 1;
            Top_Method(r) = methods(m);
            Top_Observation(r) = i;
            Top_Sentence(r) = sentences(i);
            Top_TrueLabel(r) = labels(i);
            Top_RankingType(r) = "real_tokens";
            Top_Rank(r) = k;
            Top_TokenPosition(r) = t;
            Top_Token(r) = token_text(i,t);
            Top_ImportanceCombined(r) = imp_comb(i,t,m);
            Top_ImportanceCombinedPct(r) = imp_comb_pct(i,t,m);
            Top_ImportanceOriginPct(r) = imp_origin_pct(i,t,m);
            Top_ImportanceDestinationPct(r) = imp_dest_pct(i,t,m);
        end
    end
end

% Quitar filas vacias si TOP_K era mayor que el numero de tokens reales.
valid_top = Top_Method ~= "";

T_top_tokens = table(Top_Method(valid_top), Top_Observation(valid_top), ...
    Top_Sentence(valid_top), Top_TrueLabel(valid_top), Top_RankingType(valid_top), ...
    Top_Rank(valid_top), Top_TokenPosition(valid_top), Top_Token(valid_top), ...
    Top_ImportanceCombined(valid_top), Top_ImportanceCombinedPct(valid_top), ...
    Top_ImportanceOriginPct(valid_top), Top_ImportanceDestinationPct(valid_top), ...
    'VariableNames', {'Method','Observation','Sentence','TrueLabel','RankingType', ...
    'Rank','TokenPosition','Token','ImportanceCombined','ImportanceCombinedPct', ...
    'ImportanceOriginPct','ImportanceDestinationPct'});

writetable(T_top_tokens, fullfile(OUT_DIR, 'top_tokens_por_frase_omeda_plsda_asca.csv'));

%% =========================================================
% 11) IMPORTANCIA GLOBAL POR PALABRA REAL
% ==========================================================
% Similar al analisis global de SHAP, pero ahora usando los vectores
% individuales de cada metodo. Se excluyen [CLS], [SEP] y el punto.
% ==========================================================

all_real_tokens = reshape(token_text(:,2:6), [], 1);%Extrae solo las palabras reales de todas las frases. del token 2 al 6
unique_real_tokens = unique(all_real_tokens);

n_rows_global = numel(methods) * numel(unique_real_tokens);%Calcula filas de la tabla global.

Global_Method = strings(n_rows_global,1);
Global_Token = strings(n_rows_global,1);
Global_MeanImportancePct = zeros(n_rows_global,1);
Global_MaxImportancePct = zeros(n_rows_global,1);
Global_Count = zeros(n_rows_global,1);

r = 0;
for m = 1:numel(methods)%recorre metodos
    vals_pct = squeeze(imp_comb_pct(:,:,m));%Obtiene la matriz de importancia combinada porcentual

    for u = 1:numel(unique_real_tokens)%Recorre cada palabra única.
        tok = unique_real_tokens(u);%Selecciona una palabra.
        mask_tok = token_text == tok;%Busca dónde aparece esa palabra en las 40 frases.
        token_values = vals_pct(mask_tok);%Extrae sus valores de importancia.

        r = r + 1;
        Global_Method(r) = methods(m);
        Global_Token(r) = tok;
        Global_MeanImportancePct(r) = mean(token_values);%Calcula la importancia media de esa palabra.
        Global_MaxImportancePct(r) = max(token_values);
        Global_Count(r) = numel(token_values);%Cuenta cuántas veces aparece esa palabra.
    end
end

T_global_tokens = table(Global_Method, Global_Token, Global_MeanImportancePct, ...
    Global_MaxImportancePct, Global_Count, ...
    'VariableNames', {'Method','Token','MeanImportancePct','MaxImportancePct','Count'});
%Este archivo es el equivalente a tu gráfico SHAP global
writetable(T_global_tokens, fullfile(OUT_DIR, 'global_top_tokens_reales_omeda_plsda_asca.csv'));

%% =========================================================
% 12) FIGURAS GLOBALES TIPO SHAP, UNA POR METODO
% ==========================================================

TOP_GLOBAL = 20;

for m = 1:numel(methods)
    idx_m = T_global_tokens.Method == methods(m);
    Tm = T_global_tokens(idx_m, :);
    Tm = sortrows(Tm, 'MeanImportancePct', 'descend');
    Tm = Tm(1:min(TOP_GLOBAL, height(Tm)), :);

    vals = flipud(Tm.MeanImportancePct);
    toks = flipud(Tm.Token);

    figure('Color','w','Position',[100 100 1200 750]);
    barh(vals);
    yticks(1:numel(toks));
    yticklabels(toks);
    xlabel('Importancia media relativa (%)','FontSize',14,'FontWeight','bold');
    ylabel('Token','FontSize',14,'FontWeight','bold');
    title(sprintf('Tokens mas importantes segun %s', methods(m)), ...
        'FontSize',16,'FontWeight','bold');
    grid on;
    set(gca,'FontSize',13,'LineWidth',1.2);
    set(gcf,'PaperPositionMode','auto');

    safe_name = lower(strrep(methods(m), '-', '_'));
    safe_name = strrep(safe_name, ' ', '_');
    exportgraphics(gcf, fullfile(FIG_DIR, sprintf('global_top_tokens_%s.png', safe_name)), 'Resolution', 300);
end

%% =========================================================
% 13) FIGURAS INDIVIDUALES OPCIONALES
% ==========================================================

if PLOT_INDIVIDUAL
    for m = 1:numel(methods)
        safe_method = lower(strrep(methods(m), '-', '_'));
        safe_method = strrep(safe_method, ' ', '_');

        for i = 1:n_obs
            values = squeeze(imp_comb_pct(i,:,m));

            figure('Color','w','Position',[100 100 1000 600]);
            bar(values);
            xticks(1:n_tokens);
            xticklabels(token_text(i,:));
            xlabel('Token','FontSize',14,'FontWeight','bold');
            ylabel('Importancia relativa combinada (%)','FontSize',14,'FontWeight','bold');
            title(sprintf('%s - Frase %02d - %s', methods(m), i, labels(i)), ...
                'FontSize',15,'FontWeight','bold');
            grid on;
            set(gca,'FontSize',12,'LineWidth',1.2);
            exportgraphics(gcf, fullfile(INDIV_FIG_DIR, sprintf('%s_frase_%02d.png', safe_method, i)), 'Resolution', 300);
            close(gcf);
        end
    end
end

%% =========================================================
% 14) GUARDAR RESULTADOS MATLAB
% ==========================================================

save(fullfile(OUT_DIR, 'analisis_individualizado_tokens_toolbox.mat'), ...
    'V_omeda', 'V_plsda', 'V_asca', ...
    'imp_origin', 'imp_dest', 'imp_comb', ...
    'imp_origin_pct', 'imp_dest_pct', 'imp_comb_pct', ...
    'T_all_tokens', 'T_top_tokens', 'T_global_tokens', ...
    'token_text', 'sentences', 'labels', 'Y', 'class_id', 'pair_id', ...
    'pcs_omeda', 'lvs_pls', 'bestLV', 'tabla_anova_asca', 'parglmo', 'ascao');

fprintf('\nAnalisis individualizado finalizado.\n');
fprintf('Resultados guardados en la carpeta: %s\n', OUT_DIR);
fprintf('Archivo principal: top_tokens_por_frase_omeda_plsda_asca.csv\n');
