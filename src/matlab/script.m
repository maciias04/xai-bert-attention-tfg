clear
close all
clc

data = importdata('attention_matrix_40x9216.csv');

X = data.data;

class = [repmat("Positivas",20,1);repmat("Negativas",20,1)];
y = [ones(20,1);2*ones(20,1)];

tokens = [];
for i=1:8, tokens = [tokens;sprintf("Token %d",i)]; end

attention = [];
for i=1:12, attention = [attention;sprintf("Attention %d",i)]; end

layer = [];
for i=1:12, layer = [layer;sprintf("Layer %d",i)]; end

tokenO = repmat(tokens,1, 8,12,12);
tokenO = tokenO(:);

tokenD = repmat(tokens',8, 1,12,12);
tokenD = tokenD(:);

att = repmat(attention',8*8, 1,12);
att = att(:);

lay = repmat(layer',8*8*12, 1);
lay = lay(:);

%% ASCA

par = repmat(1:20,1,2)';
[t,struct] = parglm(X,[class par]);

  ascao = asca(struct);
  
  i=1;
scores(ascao.factors{i},'Title',sprintf('Factor %d',i),'ObsClass',ascao.design(:,i));
loadings(ascao.factors{i},'Title',sprintf('Factor %d',i));


i=2;
ascao.factors{i}.lvs=1:2;
scores(ascao.factors{i},'Title',sprintf('Factor %d',i),'ObsClass',ascao.design(:,i));
loadings(ascao.factors{i},'Title',sprintf('Factor %d',i));

%% =========================================================
%  CONDENSACIÓN ASCA A NIVEL DE TOKEN
% ==========================================================
% Se trabaja directamente con los nombres ya definidos:
%   X, class, tokens, attention, layer, tokenO, tokenD, att, lay
%
% Objetivo:
%   - Comparar Positivas vs Negativas
%   - Pasar de 9216 variables a una explicación por:
%       1) token origen
%       2) token destino
%       3) pares token origen -> token destino
%       4) layer
%       5) attention/head
%       6) layer-attention
% ==========================================================

[n_obs, n_vars] = size(X);

idx_pos = class == "Positivas";
idx_neg = class == "Negativas";

Xpos = X(idx_pos,:);
Xneg = X(idx_neg,:);

% Diferencia media entre clases
% Valor positivo  -> mayor atención media en frases positivas
% Valor negativo  -> mayor atención media en frases negativas
delta_vec = mean(Xpos,1) - mean(Xneg,1);

% Importancia sin signo
importance_vec = abs(delta_vec);

% Tabla con la interpretación de cada una de las 9216 variables
tabla_variables = table((1:n_vars)', tokenO, tokenD, att, lay, ...
    delta_vec(:), importance_vec(:), ...
    'VariableNames', {'Variable', 'TokenOrigen', 'TokenDestino', ...
    'Attention', 'Layer', 'Efecto', 'Importancia'});

disp('Primeras variables interpretadas:');
disp(tabla_variables(1:10,:));

%% =========================================================
%  IMPORTANCIA POR TOKEN ORIGEN
% ==========================================================

imp_tokenO = zeros(8,1);
eff_tokenO = zeros(8,1);

for i = 1:8
    idx = tabla_variables.TokenOrigen == tokens(i);
    imp_tokenO(i) = sum(tabla_variables.Importancia(idx));
    eff_tokenO(i) = sum(tabla_variables.Efecto(idx));
end

imp_tokenO_pct = 100 * imp_tokenO / sum(imp_tokenO);

tabla_tokenO = table(tokens, imp_tokenO_pct, eff_tokenO, ...
    'VariableNames', {'TokenOrigen', 'ImportanciaPorcentaje', 'EfectoConSigno'});

disp('Importancia por token origen:');
disp(tabla_tokenO);

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.84 0.72];

bar(ax, imp_tokenO_pct);

xticks(ax, 1:8);
xticklabels(ax, tokens);

xlabel(ax, 'Token origen', 'FontSize', 18, 'FontWeight', 'bold');
ylabel(ax, 'Importancia relativa (%)', 'FontSize', 18, 'FontWeight', 'bold');

sgtitle('ASCA: importancia por token origen', ...
    'FontSize', 22, 'FontWeight', 'bold');

set(ax, 'FontSize', 15, 'LineWidth', 1.2);
grid(ax, 'on');

saveas(gcf, 'asca_importancia_token_origen.png');

%% =========================================================
%  IMPORTANCIA POR TOKEN DESTINO
% ==========================================================

imp_tokenD = zeros(8,1);
eff_tokenD = zeros(8,1);

for i = 1:8
    idx = tabla_variables.TokenDestino == tokens(i);
    imp_tokenD(i) = sum(tabla_variables.Importancia(idx));
    eff_tokenD(i) = sum(tabla_variables.Efecto(idx));
end

imp_tokenD_pct = 100 * imp_tokenD / sum(imp_tokenD);

tabla_tokenD = table(tokens, imp_tokenD_pct, eff_tokenD, ...
    'VariableNames', {'TokenDestino', 'ImportanciaPorcentaje', 'EfectoConSigno'});

disp('Importancia por token destino:');
disp(tabla_tokenD);

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.84 0.72];

bar(ax, imp_tokenD_pct);

xticks(ax, 1:8);
xticklabels(ax, tokens);

xlabel(ax, 'Token destino', 'FontSize', 18, 'FontWeight', 'bold');
ylabel(ax, 'Importancia relativa (%)', 'FontSize', 18, 'FontWeight', 'bold');

sgtitle('ASCA: importancia por token destino', ...
    'FontSize', 22, 'FontWeight', 'bold');

set(ax, 'FontSize', 15, 'LineWidth', 1.2);
grid(ax, 'on');

saveas(gcf, 'asca_importancia_token_destino.png');
%% =========================================================
%  MAPA TOKEN ORIGEN -> TOKEN DESTINO
% ==========================================================

map_token_signed = zeros(8,8);
map_token_importance = zeros(8,8);

for i = 1:8
    for j = 1:8
        idx = tabla_variables.TokenOrigen == tokens(i) & ...
              tabla_variables.TokenDestino == tokens(j);

        map_token_signed(i,j) = sum(tabla_variables.Efecto(idx));
        map_token_importance(i,j) = sum(tabla_variables.Importancia(idx));
    end
end

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.12 0.15 0.72 0.70];

imagesc(ax, map_token_signed);

cb = colorbar(ax);
cb.FontSize = 15;
cb.Label.String = 'Diferencia de atención: Positivas - Negativas';
cb.Label.FontSize = 18;
cb.Label.FontWeight = 'bold';

xticks(ax, 1:8);
yticks(ax, 1:8);

xticklabels(ax, tokens);
yticklabels(ax, tokens);

xlabel(ax, 'Token destino', 'FontSize', 18, 'FontWeight', 'bold');
ylabel(ax, 'Token origen', 'FontSize', 18, 'FontWeight', 'bold');

sgtitle('ASCA: diferencia de atención por pares de tokens', ...
    'FontSize', 22, 'FontWeight', 'bold');

set(ax, 'FontSize', 15, 'LineWidth', 1.2);

saveas(gcf, 'asca_mapa_token_origen_destino.png');
%% =========================================================
%  IMPORTANCIA POR LAYER
% ==========================================================

imp_layer = zeros(12,1);
eff_layer = zeros(12,1);

for i = 1:12
    idx = tabla_variables.Layer == layer(i);
    imp_layer(i) = sum(tabla_variables.Importancia(idx));
    eff_layer(i) = sum(tabla_variables.Efecto(idx));
end

imp_layer_pct = 100 * imp_layer / sum(imp_layer);

tabla_layer = table(layer, imp_layer_pct, eff_layer, ...
    'VariableNames', {'Layer', 'ImportanciaPorcentaje', 'EfectoConSigno'});

disp('Importancia por layer:');
disp(tabla_layer);

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.84 0.72];

bar(ax, imp_layer_pct);

xticks(ax, 1:12);
xticklabels(ax, layer);

xlabel(ax, 'Layer', 'FontSize', 18, 'FontWeight', 'bold');
ylabel(ax, 'Importancia relativa (%)', 'FontSize', 18, 'FontWeight', 'bold');

sgtitle('ASCA: importancia por layer', ...
    'FontSize', 22, 'FontWeight', 'bold');

set(ax, 'FontSize', 15, 'LineWidth', 1.2);
grid(ax, 'on');

saveas(gcf, 'asca_importancia_layer.png');
%% =========================================================
%  IMPORTANCIA POR ATTENTION
% ==========================================================

imp_attention = zeros(12,1);
eff_attention = zeros(12,1);

for i = 1:12
    idx = tabla_variables.Attention == attention(i);
    imp_attention(i) = sum(tabla_variables.Importancia(idx));
    eff_attention(i) = sum(tabla_variables.Efecto(idx));
end

imp_attention_pct = 100 * imp_attention / sum(imp_attention);

tabla_attention = table(attention, imp_attention_pct, eff_attention, ...
    'VariableNames', {'Attention', 'ImportanciaPorcentaje', 'EfectoConSigno'});

disp('Importancia por attention:');
disp(tabla_attention);

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.84 0.72];

bar(ax, imp_attention_pct);

xticks(ax, 1:12);
xticklabels(ax, attention);

xlabel(ax, 'Attention', 'FontSize', 18, 'FontWeight', 'bold');
ylabel(ax, 'Importancia relativa (%)', 'FontSize', 18, 'FontWeight', 'bold');

sgtitle('ASCA: importancia por attention', ...
    'FontSize', 22, 'FontWeight', 'bold');

set(ax, 'FontSize', 15, 'LineWidth', 1.2);
grid(ax, 'on');

saveas(gcf, 'asca_importancia_attention.png');
%% =========================================================
%  MAPA LAYER - ATTENTION
% ==========================================================

map_layer_attention = zeros(12,12);

for i = 1:12
    for j = 1:12
        idx = tabla_variables.Layer == layer(i) & ...
              tabla_variables.Attention == attention(j);

        map_layer_attention(i,j) = sum(tabla_variables.Importancia(idx));
    end
end

figure('Color','w','Position',[100 100 1200 700]);

ax = axes;
ax.Position = [0.10 0.14 0.76 0.72];

imagesc(ax, map_layer_attention);

cb = colorbar(ax);
cb.FontSize = 15;
cb.Label.String = 'Importancia acumulada';
cb.Label.FontSize = 18;
cb.Label.FontWeight = 'bold';

xticks(ax, 1:12);
yticks(ax, 1:12);

xticklabels(ax, attention);
yticklabels(ax, layer);

xlabel(ax, 'Attention', 'FontSize', 18, 'FontWeight', 'bold');
ylabel(ax, 'Layer', 'FontSize', 18, 'FontWeight', 'bold');

sgtitle('ASCA: importancia por layer y attention', ...
    'FontSize', 22, 'FontWeight', 'bold');

set(ax, 'FontSize', 15, 'LineWidth', 1.2);

saveas(gcf, 'asca_mapa_layer_attention.png');
%% =========================================================
%  IMPORTANCIA COMBINADA POR TOKEN EN ASCA
% ==========================================================
% Combinamos importancia como origen + importancia como destino
% para obtener una medida comparable con SHAP.
% ==========================================================

imp_token_comb_asca = imp_tokenO + imp_tokenD;

imp_token_comb_asca_pct = 100 * imp_token_comb_asca / sum(imp_token_comb_asca);

tabla_asca_tokens = table(tokens, ...
    imp_tokenO_pct, ...
    imp_tokenD_pct, ...
    imp_token_comb_asca_pct, ...
    'VariableNames', {'Token','Origen','Destino','Combinada'});

writetable(tabla_asca_tokens, 'asca_importancia_tokens.csv');

figure('Color','w','Position',[100 100 1200 700]);

bar([imp_tokenO_pct, imp_tokenD_pct, imp_token_comb_asca_pct]);

xticks(1:8);
xticklabels(tokens);

xlabel('Token', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Importancia relativa (%)', 'FontSize', 18, 'FontWeight', 'bold');

legend({'Como origen','Como destino','Combinada'}, ...
       'Location','best', ...
       'FontSize', 14);

title('ASCA: importancia relativa por token', ...
      'FontSize', 22, ...
      'FontWeight', 'bold');

grid on;
set(gca, 'FontSize', 15, 'LineWidth', 1.2);

exportgraphics(gcf, 'asca_importancia_tokens.png', 'Resolution', 300);