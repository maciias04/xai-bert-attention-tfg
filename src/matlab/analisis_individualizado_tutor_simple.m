%% =========================================================
%  COMPROBACION SIMPLE DEL ANALISIS INDIVIDUALIZADO
%  siguiendo literalmente el esquema propuesto por el tutor
%
%  Genera 3 graficas finales comparables:
%    - global_top_tokens_omeda_pca_tutor.png
%    - global_top_tokens_pls_da_tutor.png
%    - global_top_tokens_asca_tutor.png
%
%  Entrada esperada:
%    pca_attention_results_meda.mat
%      X          -> matriz 40 x 9216
%      sentences  -> frases usadas
%      true_label -> etiquetas positive/negative
%
%  Idea:
%    oMEDA : vecF = omedaPca(X, [], X(i,:), 1, 'Preprocessing', 1)
%    PLS   : vecF = omedaPls(X, Y, [], X(i,:), 1)
%    ASCA  : X2 = X - model.residuals;
%            vecF = omedaPca(X2, [], X2(i,:), 1)
% ==========================================================

clear;
clc;
close all;

%% =========================================================
% 0) CONFIGURACION
% ==========================================================

% Cambia esta ruta por la ruta real de la MEDA Toolbox en tu ordenador.
toolbox_path = 'C:\Users\Usuario\Desktop\UNI\Cuarto de Carrera Teleco\doctoradopepe\MEDA Shared Materials-20260301T114916Z-3-001\MEDA Shared Materials\MEDA-Toolbox-master\MEDA-Toolbox-master\toolbox';

if exist(toolbox_path, 'dir')
    addpath(genpath(toolbox_path));
else
    warning('No se ha encontrado toolbox_path. Si la toolbox ya esta en el path, puedes ignorar este aviso.');
end

DATA_FILE = 'pca_attention_results_meda.mat';
OUT_DIR   = 'resultados_tutor_simple';
FIG_DIR   = fullfile(OUT_DIR, 'figuras');

if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end
if ~exist(FIG_DIR, 'dir'), mkdir(FIG_DIR); end

% Para oMEDA-PCA uso lo que puso el tutor en su ejemplo: Preprocessing = 1.
PREP_OMEDA = 2;

% Numero de tokens del experimento:
% [CLS] + 5 palabras + . + [SEP] = 8
n_tokens = 8;

% Numero de palabras reales que se muestran en las graficas globales.
TOP_GLOBAL = 20;

%% =========================================================
% 1) COMPROBAR FUNCIONES DE LA TOOLBOX
% ==========================================================

required_functions = {'omedaPca','omedaPls','parglm','asca'};

for k = 1:numel(required_functions)
    if isempty(which(required_functions{k}))
        error('No se encuentra la funcion %s. Revisa que la MEDA Toolbox este en el path.', required_functions{k});
    end
end

disp('Funciones necesarias de la MEDA Toolbox encontradas.');

%% =========================================================
% 2) CARGAR DATOS
% ==========================================================

if ~exist(DATA_FILE, 'file')
    error('No se encuentra %s. Pon este script en la misma carpeta que el archivo .mat.', DATA_FILE);
end

load(DATA_FILE);

if ~exist('X', 'var')
    error('El archivo %s no contiene la matriz X.', DATA_FILE);
end

if ~exist('sentences', 'var')
    error('El archivo %s no contiene la variable sentences.', DATA_FILE);
end

X = double(X);
sentences = string(sentences(:));

[n_obs, n_vars] = size(X);

fprintf('Tamaño de X: %d x %d\n', n_obs, n_vars);

if n_obs ~= 40
    error('Este script esta preparado para 40 frases. Observaciones encontradas: %d.', n_obs);
end

if n_vars ~= 12*12*8*8
    error('Se esperaban 9216 variables = 12*12*8*8. Variables encontradas: %d.', n_vars);
end

%% =========================================================
% 3) DEFINIR Y Y F EXACTAMENTE COMO EN EL ESQUEMA DEL TUTOR
% ==========================================================

% 20 frases positivas seguidas de 20 negativas.
Y = [ones(20,1); -ones(20,1)];%Crea la variable de clase

% Factor clase + factor pareja.
% Pareja 1: frase 1 positiva con frase 21 negativa, etc.
F = [Y, [1:20 1:20]'];

%% =========================================================
% 4) TOKENIZACION SIMPLE PARA ETIQUETAR LAS GRAFICAS
% ==========================================================

% token_text(i,:) contiene:
% [CLS] palabra1 palabra2 palabra3 palabra4 palabra5 . [SEP]
token_text = strings(n_obs, n_tokens);

for i = 1:n_obs
    s = lower(strtrim(sentences(i)));
    s = replace(s, '.', ' .');
    parts = split(s);
    parts(parts == '') = [];

    if numel(parts) == 6
        token_text(i,:) = ["[CLS]", reshape(parts, 1, []), "[SEP]"];
    else
        warning('La frase %d no tiene 5 palabras + punto. Se usan etiquetas genericas.', i);
        token_text(i,:) = ["[CLS]", "Token 2", "Token 3", "Token 4", "Token 5", "Token 6", ".", "[SEP]"];
    end
end

%% =========================================================
% 5) MAPEAR CADA VARIABLE A TOKEN ORIGEN Y TOKEN DESTINO
% ==========================================================

% Orden asumido de variables:
% layer -> head -> token origen -> token destino
var_index = (1:n_vars)';
zero_idx = var_index - 1;

n_layers = 12;
n_heads  = 12;

rem_layer = mod(zero_idx, n_heads*n_tokens*n_tokens);
rem_head  = mod(rem_layer, n_tokens*n_tokens);

tokenO_idx = floor(rem_head / n_tokens) + 1;
tokenD_idx = mod(rem_head, n_tokens) + 1;

%% =========================================================
% 6) oMEDA-PCA TAL COMO PROPONE EL TUTOR
% ==========================================================

fprintf('\nCalculando oMEDA-PCA segun esquema del tutor...\n');

% Global positivo frente a negativo.
vecX_omeda = omedaPca(X, [], X, Y, 'Preprocessing', PREP_OMEDA);

% Individual frase a frase.
V_omeda = zeros(n_obs, n_vars);

for i = 1:n_obs
    vecF = omedaPca(X, [], X(i,:), 1, 'Preprocessing', PREP_OMEDA);
    V_omeda(i,:) = vecF(:)';
end

%% =========================================================
% 7) PLS-DA / oMEDA-PLS TAL COMO PROPONE EL TUTOR
% ==========================================================

fprintf('Calculando PLS-DA / oMEDA-PLS segun esquema del tutor...\n');

% Modelo PLS sencillo con una variable latente, tal como indico el tutor.
lvs = 1;
Xcs = preprocess2D(X);
Ycs = preprocess2D(Y);
model_pls = simpls(Xcs, Ycs, 'LVs', lvs);

% Salida global del modelo PLS.
% En algunas versiones de la MEDA Toolbox el campo se llama pred,
% pero en otras versiones no existe. Para este script no se usa en
% las graficas finales; se guarda solo como comprobacion.
if isfield(model_pls, 'pred')
    vecX_pls_pred = model_pls.pred;
elseif isfield(model_pls, 'beta')
    vecX_pls_pred = Xcs * model_pls.beta;
else
    warning('El modelo PLS no contiene ni pred ni beta. Se omite vecX_pls_pred.');
    vecX_pls_pred = NaN(n_obs,1);
end
% Individual frase a frase con omedaPls directamente.
V_pls = zeros(n_obs, n_vars);

for i = 1:n_obs
    vecF = omedaPls(X, Y, [], X(i,:), 1);
    V_pls(i,:) = vecF(:)';
end

%% =========================================================
% 8) ASCA TAL COMO PROPONE EL TUTOR
% ==========================================================

fprintf('Calculando ASCA segun esquema del tutor...\n');

[tabla_asca, struct_asca] = parglm(X, F);
model_asca = asca(struct_asca);

% Carga del factor clase.
vecX_asca_loads = model_asca.factors{1}.loads;

% Matriz ajustada segun el esquema del tutor.
X2 = X - model_asca.residuals;

% Individual frase a frase sobre X2.
V_asca = zeros(n_obs, n_vars);

for i = 1:n_obs
    vecF = omedaPca(X2, [], X2(i,:), 1);
    V_asca(i,:) = vecF(:)';
end

%% =========================================================
% 9) PASAR DE 9216 VARIABLES A IMPORTANCIA POR TOKEN
% ==========================================================

% Se usa valor absoluto porque queremos magnitud de importancia.
% Para cada frase:
%   importancia token = suma como origen + suma como destino.

methods = ["oMEDA-PCA", "PLS-DA", "ASCA"];
V_all = cat(3, V_omeda, V_pls, V_asca);

imp_comb_pct = zeros(n_obs, n_tokens, numel(methods));

for m = 1:numel(methods)
    for i = 1:n_obs
        v_abs = abs(V_all(i,:,m))';

        imp_origin = accumarray(tokenO_idx, v_abs, [n_tokens 1], @sum, 0);
        imp_dest   = accumarray(tokenD_idx, v_abs, [n_tokens 1], @sum, 0);
        imp_comb   = imp_origin + imp_dest;

        if sum(imp_comb) > 0
            imp_comb_pct(i,:,m) = 100 * imp_comb' / sum(imp_comb);
        end
    end
end

%% =========================================================
% 10) TABLA GLOBAL DE TOKENS REALES
% ==========================================================

% Solo palabras reales: posiciones 2 a 6.
% Se excluyen [CLS], punto y [SEP] para que sea comparable con SHAP global.
real_positions = 2:6;
all_real_tokens = reshape(token_text(:, real_positions), [], 1);
unique_real_tokens = unique(all_real_tokens);

rows = [];

Method = strings(numel(methods)*numel(unique_real_tokens),1);
Token = strings(numel(methods)*numel(unique_real_tokens),1);
MeanImportancePct = zeros(numel(methods)*numel(unique_real_tokens),1);
MaxImportancePct = zeros(numel(methods)*numel(unique_real_tokens),1);
Count = zeros(numel(methods)*numel(unique_real_tokens),1);

r = 0;

for m = 1:numel(methods)
    vals_pct = squeeze(imp_comb_pct(:,:,m));

    for u = 1:numel(unique_real_tokens)
        tok = unique_real_tokens(u);
        mask_tok = token_text == tok;
        token_values = vals_pct(mask_tok);

        if isempty(token_values)
            continue;
        end

        r = r + 1;
        Method(r) = methods(m);
        Token(r) = tok;
        MeanImportancePct(r) = mean(token_values);
        MaxImportancePct(r) = max(token_values);
        Count(r) = numel(token_values);
    end
end

T_global = table(Method(1:r), Token(1:r), MeanImportancePct(1:r), MaxImportancePct(1:r), Count(1:r), ...
    'VariableNames', {'Method','Token','MeanImportancePct','MaxImportancePct','Count'});

writetable(T_global, fullfile(OUT_DIR, 'global_top_tokens_tutor_simple.csv'));

%% =========================================================
% 11) GENERAR LAS TRES GRAFICAS FINALES
% ==========================================================

for m = 1:numel(methods)
    Tm = T_global(T_global.Method == methods(m), :);
    Tm = sortrows(Tm, 'MeanImportancePct', 'descend');
    Tm = Tm(1:min(TOP_GLOBAL, height(Tm)), :);

    vals = flipud(Tm.MeanImportancePct);
    toks = flipud(Tm.Token);

    figure('Color','w','Position',[100 100 1200 750]);
    barh(vals);
    yticks(1:numel(toks));
    yticklabels(toks);
    xlabel('Importancia media relativa (%)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Token', 'FontSize', 14, 'FontWeight', 'bold');
    title(sprintf('Tokens mas importantes segun %s', methods(m)), ...
        'FontSize', 16, 'FontWeight', 'bold');
    set(gca, 'FontSize', 12, 'LineWidth', 1.2);
    grid on;
    box on;
    set(gca, 'YDir', 'normal');
    tightfig_if_available();

    if methods(m) == "oMEDA-PCA"
        out_name = 'global_top_tokens_omeda_pca_tutor.png';
    elseif methods(m) == "PLS-DA"
        out_name = 'global_top_tokens_pls_da_tutor.png';
    else
        out_name = 'global_top_tokens_asca_tutor.png';
    end

    exportgraphics(gcf, fullfile(FIG_DIR, out_name), 'Resolution', 300);
end

%% =========================================================
% 12) GUARDAR TODO PARA COMPARAR
% ==========================================================

save(fullfile(OUT_DIR, 'resultados_tutor_simple.mat'), ...
    'X', 'Y', 'F', 'sentences', 'token_text', ...
    'vecX_omeda', 'V_omeda', ...
    'model_pls', 'vecX_pls_pred', 'V_pls', ...
    'tabla_asca', 'struct_asca', 'model_asca', 'vecX_asca_loads', 'X2', 'V_asca', ...
    'imp_comb_pct', 'T_global', 'tokenO_idx', 'tokenD_idx');

fprintf('\nAnalisis finalizado.\n');
fprintf('Resultados en: %s\n', OUT_DIR);
fprintf('Figuras en: %s\n', FIG_DIR);

%% =========================================================
% Funcion pequena solo para evitar error si tightfig no esta instalado
% ==========================================================
function tightfig_if_available()
    if exist('tightfig', 'file')
        tightfig;
    else
        drawnow;
    end
end
