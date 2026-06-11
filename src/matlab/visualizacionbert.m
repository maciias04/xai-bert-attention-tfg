%% ============================================================
%  Visualización de la estructura completa de BERT en MATLAB
%  Objetivo:
%  - Cargar un modelo BERT-base
%  - Ver la arquitectura completa
%  - Guardar una imagen de la estructura
% ============================================================

clear;
clc;
close all;

%% ============================================================
% 1) CARGAR MODELO BERT
% ============================================================

% Este comando carga el modelo BERT-base disponible en MATLAB.
% Normalmente requiere Text Analytics Toolbox y/o el soporte de modelos
% preentrenados correspondiente.
try
    modeloBERT = bert;
catch ME
    error("No se ha podido cargar BERT. Revisa que tengas instalado el soporte de BERT en MATLAB. Error: %s", ME.message);
end

%% ============================================================
% 2) EXTRAER LA RED DEL MODELO
% ============================================================

% Dependiendo de la versión de MATLAB, bert puede devolver directamente
% una red o un objeto que contiene la red en la propiedad Network.

if isa(modeloBERT, "dlnetwork") || isa(modeloBERT, "DAGNetwork") || isa(modeloBERT, "SeriesNetwork")
    net = modeloBERT;
elseif isprop(modeloBERT, "Network")
    net = modeloBERT.Network;
elseif isfield(modeloBERT, "Network")
    net = modeloBERT.Network;
else
    error("No se ha encontrado la red dentro del objeto BERT cargado.");
end

disp("Modelo BERT cargado correctamente.");
disp(net);

%% ============================================================
% 3) ABRIR VISUALIZACIÓN INTERACTIVA DE MATLAB
% ============================================================

% Esta es la forma más cómoda para explorar la red completa.
analyzeNetwork(net);

%% ============================================================
% 4) ABRIR EN DEEP NETWORK DESIGNER
% ============================================================

% Esta opción permite ver la red de forma gráfica e interactiva.
% Si tu versión de MATLAB no lo soporta, puedes comentar esta línea.
try
    deepNetworkDesigner(net);
catch
    warning("No se ha podido abrir Deep Network Designer con este objeto de red.");
end

%% ============================================================
% 5) INTENTAR GRAFICAR Y GUARDAR LA ESTRUCTURA COMPLETA
% ============================================================

% Convertir a layerGraph si es posible.
try
    lgraph = layerGraph(net);
catch
    lgraph = net;
end

fig = figure;
fig.Position = [100 100 1400 2200];

try
    plot(lgraph);
    title("Estructura completa del modelo BERT", "FontSize", 16, "FontWeight", "bold");
catch
    warning("No se ha podido representar la red con plot(layerGraph). Usa analyzeNetwork.");
end

%% ============================================================
% 6) GUARDAR LA FIGURA
% ============================================================

% Crear carpeta de salida
outputFolder = "figuras_bert";

if ~exist(outputFolder, "dir")
    mkdir(outputFolder);
end

% Guardar como PNG
try
    exportgraphics(fig, fullfile(outputFolder, "estructura_completa_bert.png"), "Resolution", 300);
    disp("Imagen guardada en: figuras_bert/estructura_completa_bert.png");
catch
    warning("No se ha podido exportar la figura como PNG.");
end

% Guardar también como PDF
try
    exportgraphics(fig, fullfile(outputFolder, "estructura_completa_bert.pdf"), "ContentType", "vector");
    disp("PDF guardado en: figuras_bert/estructura_completa_bert.pdf");
catch
    warning("No se ha podido exportar la figura como PDF.");
end