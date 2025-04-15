function Template_T5_E1
% Este script contiene la resolución del ejercicio aplicado 1 del Tema 5
% de la asignatura 'Técnicas de Inteligencia Artificial'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Este ejercicio está relacionado con el uso de la base de datos College. 
% Trataremos de predecir el número de solicitudes de mátriculación (Apps)
% recibidas usando las otras variables de la base de datos.

clear all;
clc;
close all;

% Cargamos base de datos
load('College');
College = rmmissing(College);
var_names = College.Properties.VariableNames

disp('%%%%%%%%%%%%%%%%% EJERCICIO 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')

disp('%%%%%%%%%%%%%%%%% Apartado 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 1 - Divide los datos de manera aleatoria en conjuntos de 
% entrenamiento (50%) y de test (50%). Fijar la semilla para la 
% generación de números pseudo-aleatorios a rng(13).

rng(13); % Fijamos semilla para el generado de números aleatorios

collegeHalves = cvpartition(size(College, 1), "holdOut", 0.5);
pos_train = collegeHalves.training;
pos_test = collegeHalves.test;

% Binarizamos Private
D = dummyvar(categorical(College{:,2}));
Dcells = num2cell(D(:,2));

% Separamos "Apps" del resto de datos.
AllX = [Dcells, College(:, 4:end)];
AllY = [College(:, 3)];

Xtrain = table2array(AllX(pos_train,:));
Xtest = table2array(AllX(pos_test,:));
Ytrain = table2array(AllY(pos_train,:));
Ytest = table2array(AllY(pos_test,:));

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 2 - Ajustar un modelo de regresión lineal por mínimos cuadráticos  
% usando el conjunto de entrenamiento. Reportar el error de test obtenido.

mdl = fitlm(Xtrain, Ytrain);
ypred = predict(mdl, Xtest);

MSE_test = mean((Ytest-ypred).^2);
%mean((Ytest-predict(mdl, Xtrain)).^2)
fprintf("MSE de modelo lineal: %4.4f", MSE_test)
mdl_lm = mdl;

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 3 - Ajustar un modelo ridge regression usando el conjunto de 
% entrenamiento y con una lambda seleccionada mediante 10-fold CV. 
% Reportar el error de test obtenido.

rng(3);

k = 10;

c = cvpartition(length(Ytrain), 'KFold', k);

lambda_grid = linspace(0.01,100,100);

for aa = 1:k
    % Vamos a iterar por subsecciones de la base de datos, así que tenemos que crear una por cada iteración!
    pos_train_CV =  c.training(aa);
    pos_test_CV  =  c.test(aa);

    %size(Xtrain)
    %size(pos_test_CV)

    Xtrain_iter = Xtrain(pos_train_CV,:);
    Xtest_iter = Xtrain(pos_test_CV,:);
    Ytrain_iter = Ytrain(pos_train_CV,:);
    Ytest_iter = Ytrain(pos_test_CV,:);

    % Para cada lambda, ajustar modelo y guardar su MSE para luego filtrar
    for bb=1:length(lambda_grid) 

        B = ridge(Ytrain_iter,Xtrain_iter,lambda_grid(bb),0);
        ypred = B(1) + Xtest_iter*B(2:end);
        MSE_test= mean((Ytest_iter-ypred).^2);
        CV_MSE(aa,bb) = MSE_test;
    end
end

[val, pos] = min(mean(CV_MSE));

mdl_ridge = ridge(Ytrain,zscore(Xtrain),lambda_grid(pos),0);

fprintf("MSE para lambda optima: %4.f\n", val)

figure()
plot(lambda_grid,mean(CV_MSE,1));
title('RIDGE');
hold on;
plot(lambda_grid(pos),val,'ro');
hold off;

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 4 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 4 - Ajustar un modelo lasso usando el conjunto de entrenamiento 
% y con una lambda seleccionada mediante 10-fold CV. Reportar el error de 
% test obtenido y el número de estimaciones de los coeficientes diferentes de 
% cero.

% Misma idea que el apartado 3 pero con lasso, solo cambia lo de dentro de los for iteradores

rng(3);

k = 10;

c = cvpartition(length(Ytrain), 'KFold', k);

lambda_grid = linspace(0.01,100,100);

for aa = 1:k
    % Vamos a iterar por subsecciones de la base de datos, así que tenemos que crear una por cada iteración!
    pos_train_CV =  c.training(aa);
    pos_test_CV  =  c.test(aa);

    Xtrain_iter = Xtrain(pos_train_CV,:);
    Xtest_iter = Xtrain(pos_test_CV,:);
    Ytrain_iter = Ytrain(pos_train_CV,:);
    Ytest_iter = Ytrain(pos_test_CV,:);

    % Para cada lambda, ajustar modelo y guardar su MSE para luego filtrar
    for bb=1:length(lambda_grid) 

        [B, fitInfo] = lasso(Xtrain_iter, Ytrain_iter, 'Lambda',lambda_grid(bb));
        ypred = fitInfo.Intercept + Xtest_iter*B;
        MSE_test = mean((Ytest_iter-ypred).^2);
        CV_MSE(aa, bb) = MSE_test;
    end
end

[val, pos] = min(mean(CV_MSE));

[mdl_lasso, fitInfo] = lasso(zscore(Xtrain), Ytrain, 'Lambda',lambda_grid(pos));

fprintf("MSE para lambda optima: %4.4f\n", val)
fprintf("# Estimaciones de coefficientes != 0: %4.4f\n", lambda_grid(pos))

figure()
plot(lambda_grid,mean(CV_MSE,1));
title('LASSO');
hold on;
plot(lambda_grid(pos),val,'ro');
hold off;

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 5 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 5 - Ajustar un modelo PCR usando el conjunto de entrenamiento y 
% con una M seleccionada mediante 10-fold CV. Reportar el error de test 
% obtenido y el valor de $m$ obtenido a través de 10-fold CV.

X = zscore(table2array(AllX));
Y = table2array(AllY);

X1 = X(pos_train,:);
Y1 = Y(pos_train);
X2 = X(pos_test,:);
Y2 = Y(pos_test);

% Aplicamos PCA (Principal Component Analysis)
[PCALoadings, PCAScores, PCAVar,~,explained, mu] = pca(X1);
cumsum(explained);

% Misma idea que el apartado 3 pero con lasso, solo cambia lo de dentro de los for iteradores

rng(3);

k = 10;

c = cvpartition(length(Ytrain), 'KFold', k);

clear("CV_MSE")

for aa = 1:k
    % Vamos a iterar por subsecciones de la base de datos, así que tenemos que crear una por cada iteración!
    pos_train_CV = c.training(aa);
    pos_test_CV  = c.test(aa);

    Ytrain = Y1(pos_train_CV);
    Ytest  = Y1(pos_test_CV);

    % Para cada lambda, ajustar modelo y guardar su MSE para luego filtrar
    for bb = 1:size(X,2)

        X_PCR_train = PCAScores(pos_train_CV,(1:bb));
        X_PCR_test = PCAScores(pos_test_CV,(1:bb));

        mdl = fitlm(X_PCR_train, Ytrain);
        CV_MSE(aa,bb) = mean((Ytest-predict(mdl, X_PCR_test)).^2);
    end
end

[val, pos] = min(mean(CV_MSE));

figure()
plot(mean(CV_MSE));
hold on;
plot(pos, val, 'ro');
hold off;
xlabel("M");
ylabel("MSE para 10-FOLD CV");

mdl_PCR = fitlm(PCAScores(:,1:pos),Y1);
X_PCR_test = (Xtest-mu)*PCALoadings(:,1:pos);
PCALoadings(:,1:pos);
ypred_PCR = predict(mdl_PCR, X_PCR_test);

fprintf("MSE optimo de PCR/PCA: %4.4f\n", val)


fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 6 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 6 - Comentar los resultados obtenidos. ¿Cómo de precisas son 
% nuestras predicciones del número de solicitudes recibidas? ¿Hay mucha 
% diferencia entre los errores de test reportados en los apartados anteriores?

Xtrain = table2array(AllX(pos_train,:));
Xtest = table2array(AllX(pos_test,:));
Ytrain = table2array(AllY(pos_train,:));
Ytest = table2array(AllY(pos_test,:));

ypred_lin = predict(mdl_lm, Xtest);
ypred_ridge = mdl_ridge(1) + Xtest*mdl_ridge(2:end);
ypred_lasso = fitInfo.Intercept + Xtest*mdl_lasso;


% X_PCR_test = (X2-mu)*PCALoadings(:,1:pos);
% MSE_test_PCR = mean((Y2-predict(mdl, X_PCR_test)).^2);

figure()
%size(Xtest, 1)
%size(ypred_ridge)
%size(ypred_lasso)
%size(ypred_PCR)

fprintf("Todos los modelos hacen predicciones aceptables. Comparando el MSE, el modelo lineal inicial es el que mejor lo hace.\n")
fprintf("Los las predicciones de los modelos que empean RIDGE, LASSO y PCA requieren de un factor de corrección, que se ha estimado a groso modo a ser 40000\n")

subplot(5,1,1)
stem(linspace(0,size(Xtest,1),size(Xtest,1)), Ytest)
title("Y de test originales")
subplot(5,1,2)
stem(linspace(0,size(Xtest,1), size(Xtest,1)), ypred_lin)
title("Predicción del modelo lineal")
subplot(5,1,3)
stem(linspace(0,size(Xtest,1),size(Xtest,1)), ypred_ridge/4000)
title("Predicción del modelo conseguido con RIDGE")
subplot(5,1,4)
stem(linspace(0,size(Xtest,1),size(Xtest,1)), ypred_lasso/4000)
title("Predicción del modelo conseguido con LASSO")
subplot(5,1,5)
stem(linspace(0,size(Xtest,1),size(Xtest,1)), ypred_PCR/4000)
title("Predicción del modelo conseguido con PCA/PCR")