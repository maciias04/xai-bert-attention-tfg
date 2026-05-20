% Tamaños de fuente para las figuras
FS_TITULO  = 22;
FS_EJES    = 18;
FS_TICKS   = 15;
FS_LEYENDA = 16;
%% =========================================================
% 15) oMEDA MANUAL: POSITIVOS VS NEGATIVOS
% ==========================================================
% Como MATLAB no encuentra la función omeda de la toolbox,
% implementamos el cálculo manualmente.
%
% Idea:
%   1) Se calcula la media de los scores PCA para las frases positivas.
%   2) Se calcula la media de los scores PCA para las frases negativas.
%   3) Se calcula la diferencia entre ambas medias.
%   4) Esa diferencia se proyecta al espacio original usando los loadings P.
%
% Resultado:
%   omeda_vec -> vector 9216 x 1
%
% Interpretación:
%   omeda_vec > 0  -> variable asociada al grupo positivo
%   omeda_vec < 0  -> variable asociada al grupo negativo
%   omeda_vec ≈ 0  -> variable poco relevante
% ==========================================================

idx_pos = find(labels == "positive");
idx_neg = find(labels == "negative");

fprintf('\nNúmero de frases positivas: %d\n', numel(idx_pos));
fprintf('Número de frases negativas: %d\n', numel(idx_neg));

% Media de scores por grupo
meanT_pos = mean(T(idx_pos,:), 1);
meanT_neg = mean(T(idx_neg,:), 1);

% Diferencia entre grupos en el espacio PCA
deltaT = meanT_pos - meanT_neg;

% Proyección de la diferencia al espacio original
omeda_vec = P * deltaT';

disp('Tamaño del vector oMEDA:');
disp(size(omeda_vec));   % debe ser 9216 x 1
%% =========================================================
% 16) GRÁFICO oMEDA GLOBAL
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

sgtitle('oMEDA manual: positivo - negativo', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');
%% =========================================================
% 22) IMPORTANCIA DE TODAS LAS VARIABLES (ranking completo)
% ==========================================================

% Importancia = valor absoluto (cuánto separa, sin signo)
importance = abs(omeda_vec);

% Ranking de mayor a menor
[importance_sorted, idx_sorted_all] = sort(importance, 'descend');

% Top-K configurable (por ejemplo 50)
K = 50;
topK_idx = idx_sorted_all(1:K);
topK_vals = omeda_vec(topK_idx);

fprintf('\nTop %d variables (índice y valor oMEDA):\n', K);
disp(table(topK_idx, topK_vals));

%% =========================================================
% 23) SELECCIÓN AUTOMÁTICA DE VARIABLES "RELEVANTES"
% (sin elegir 20 arbitrariamente)
% ==========================================================
% Usamos un umbral estadístico: media + 2*std del |oMEDA|
thr = mean(importance) + 2*std(importance);

relevant_idx = find(importance >= thr);
relevant_vals = omeda_vec(relevant_idx);

fprintf('Variables relevantes según umbral (|oMEDA| >= mu+2σ): %d\n', numel(relevant_idx));

% Ver algunas
disp(table(relevant_idx(1:min(20,end)), relevant_vals(1:min(20,end))));

%% =========================================================
% 24) HISTOGRAMA DE IMPORTANCIAS
% ==========================================================

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.84 0.72];

histogram(ax, importance, 50);

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

%% =========================================================
% 25) CONTRIBUCIÓN ACUMULADA
% ==========================================================

cum_imp = cumsum(importance_sorted) / sum(importance_sorted);

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.84 0.72];

plot(ax, cum_imp, 'LineWidth', 1.8);

xlabel(ax, 'Nº de variables ordenadas por importancia', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

ylabel(ax, 'Contribución acumulada', ...
       'FontSize', FS_EJES, ...
       'FontWeight', 'bold');

sgtitle('Cuántas variables explican la mayor parte de la diferencia', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');
hold(ax, 'on');

h80 = yline(ax, 0.8, '--r', '80%');
h90 = yline(ax, 0.9, '--g', '90%');

h80.FontSize = FS_LEYENDA;
h90.FontSize = FS_LEYENDA;

h80.FontWeight = 'bold';
h90.FontWeight = 'bold';

h80.LabelHorizontalAlignment = 'left';
h90.LabelHorizontalAlignment = 'left';

h80.LabelVerticalAlignment = 'bottom';
h90.LabelVerticalAlignment = 'bottom';

ylim(ax, [0 1.05]);

hold(ax, 'off');

% ¿Cuántas variables para 80% y 90%?
n80 = find(cum_imp >= 0.8, 1, 'first');
n90 = find(cum_imp >= 0.9, 1, 'first');

fprintf('Variables necesarias para 80%%: %d\n', n80);
fprintf('Variables necesarias para 90%%: %d\n', n90);

%% =========================================================
% 26) AGRUPAR IMPORTANCIA POR CAPA Y HEAD
% ==========================================================
% 9216 = 12 capas * 12 heads * 8 * 8
imp_by_layer = zeros(12,1);
imp_by_head  = zeros(12,12);

for var_idx = 1:n_vars
    zero_idx = var_idx - 1;

    layer = floor(zero_idx / (12*64));
    rem1  = mod(zero_idx, 12*64);

    head  = floor(rem1 / 64);

    % acumulamos importancia (sin signo)
    imp_by_layer(layer+1) = imp_by_layer(layer+1) + importance(var_idx);
    imp_by_head(layer+1, head+1) = imp_by_head(layer+1, head+1) + importance(var_idx);
end

%% =========================================================
% 26.1) IMPORTANCIA POR CAPA
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

sgtitle('Importancia por capa', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

grid(ax, 'on');

%% =========================================================
% 26.2) HEATMAP DE IMPORTANCIA POR CAPA Y HEAD
% ==========================================================

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.78 0.72];

imagesc(ax, imp_by_head);

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

sgtitle('Importancia por layer y head', ...
        'FontSize', FS_TITULO, ...
        'FontWeight', 'bold');

set(ax, 'FontSize', FS_TICKS, ...
        'LineWidth', 1.2);

xticks(ax, 1:12);
yticks(ax, 1:12);

grid(ax, 'off');

%% =========================================================
% 27) IMPORTANCIA POR TOKEN EN oMEDA
% ==========================================================
% Cada variable de atención depende de:
%   layer, head, token origen, token destino
%
% Para obtener importancia por token:
%   - sumamos la importancia cuando el token actúa como origen
%   - sumamos la importancia cuando el token actúa como destino
%   - combinamos ambas para obtener una importancia total por token
% ==========================================================

n_tokens = 8;
n_heads  = 12;

token_labels = ["[CLS]"; "Token 2"; "Token 3"; "Token 4"; ...
                "Token 5"; "Token 6"; "."; "[SEP]"];

imp_tokenO_omeda = zeros(n_tokens,1);
imp_tokenD_omeda = zeros(n_tokens,1);

for var_idx = 1:n_vars

    zero_idx = var_idx - 1;

    % Estructura:
    % 9216 = 12 layers * 12 heads * 8 origen * 8 destino
    layer_idx = floor(zero_idx / (n_heads*n_tokens*n_tokens));
    rem1      = mod(zero_idx, n_heads*n_tokens*n_tokens);

    head_idx  = floor(rem1 / (n_tokens*n_tokens));
    rem2      = mod(rem1, n_tokens*n_tokens);

    tokenO_idx = floor(rem2 / n_tokens) + 1;
    tokenD_idx = mod(rem2, n_tokens) + 1;

    imp_tokenO_omeda(tokenO_idx) = imp_tokenO_omeda(tokenO_idx) + importance(var_idx);
    imp_tokenD_omeda(tokenD_idx) = imp_tokenD_omeda(tokenD_idx) + importance(var_idx);
end

% Importancia combinada
imp_token_comb_omeda = imp_tokenO_omeda + imp_tokenD_omeda;

% Normalizar a porcentaje
imp_tokenO_omeda_pct    = 100 * imp_tokenO_omeda    / sum(imp_tokenO_omeda);
imp_tokenD_omeda_pct    = 100 * imp_tokenD_omeda    / sum(imp_tokenD_omeda);
imp_token_comb_omeda_pct = 100 * imp_token_comb_omeda / sum(imp_token_comb_omeda);

% Guardar tabla
tabla_omeda_tokens = table(token_labels, ...
    imp_tokenO_omeda_pct, ...
    imp_tokenD_omeda_pct, ...
    imp_token_comb_omeda_pct, ...
    'VariableNames', {'Token','Origen','Destino','Combinada'});

writetable(tabla_omeda_tokens, 'omeda_importancia_tokens.csv');

% Figura
figure('Color','w','Position',[100 100 1200 700]);

bar([imp_tokenO_omeda_pct, imp_tokenD_omeda_pct, imp_token_comb_omeda_pct]);

xticks(1:n_tokens);
xticklabels(token_labels);

xlabel('Token', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Importancia relativa (%)', 'FontSize', 18, 'FontWeight', 'bold');

legend({'Como origen','Como destino','Combinada'}, ...
       'Location','best', ...
       'FontSize', 14);

title('oMEDA: importancia relativa por token', ...
      'FontSize', 22, ...
      'FontWeight', 'bold');

grid on;
set(gca, 'FontSize', 15, 'LineWidth', 1.2);

exportgraphics(gcf, 'omeda_importancia_tokens.png', 'Resolution', 300);