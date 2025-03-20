function practica5_hands_on_ridge_lasso_TEMPLATE
% Este script contiene la resolución del tutorial práctico del Tema 5
% de la asignatura 'Técnicas de Inteligencia Artificial'

load('..\..\0_MATERIAL\Bases_de_datos\data\MAT\Hitters');
% Nombre de las variables
var_names = Hitters.Properties.VariableNames;

% Dimensiones de la base de datos original
size(Hitters)

% Remover valores NaN
Hitters = rmmissing(Hitters);

% Dimensiones de la base de datos sin valores NaN
size(Hitters)

disp('%%%%%%%%%%%%%%% RIDGE REGRESSION & THE LASSO %%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

% Binarizo las variables cualitativas League, Division y NewLeague
D = dummyvar(categorical(Hitters{:,14}));
League_N = D(:,2);
D = dummyvar(categorical(Hitters{:,15}));
Division_W = D(:,2);
D = dummyvar(categorical(Hitters{:,20}));
NewLeague_N = D(:,2);
X = [Hitters{:,1:13} League_N Division_W Hitters{:,16:18} NewLeague_N];
Y = Hitters{:,19};

% Creamos matriz de valores posible para lambda RIDGE REGRESSION


% RIDGE


grid on;
xlabel('Lambda');
ylabel('Coeficientes');
title('RIDGE REGRESSION');

% Creamos matriz de valores posible para lambda LASSO


% LASSO


grid on;
xlabel('Lambda');
ylabel('Coeficientes');
title('LASSO');
pause;close;

% Dividimos la base de datos en train y test
rng(1); % Fijamos semilla para la generación de números aleatorios

% Partición no estratificada 50% train y 50% test


% RIDGE REGRESSION
lambda = 4;




fprintf('\nMSE test (RIDGE, lambda = %.1f) = %.0f \n',lambda,MSE_test);
fprintf('MSE test del modelo nulo = %.0f \n',mean((mean(Y(pos_train))-Y(pos_test)).^2));



lambda = 10^10;




fprintf('MSE test (RIDGE, lambda = %.0f) = %.0f \n',lambda,MSE_test);




lambda = 0;



fprintf('MSE test (RIDGE, lambda = %.0f) = %.0f \n',lambda,MSE_test);

% least squares



fprintf('MSE test (least squares) = %.0f \n',MSE_test);

% Con LASSO
lambda = 4;



fprintf('\nMSE test (LASSO, lambda = %.0f) = %.0f \n',lambda,MSE_test_LASSO);




% Usamos ahora 10 FOLD CV en entrenamiento para seleccionar lambda óptima
% Creamos particiones
rng(2);
k = 10;




lambda_grid = linspace(0.01,100,100);
lambda_grid_LASSO = linspace(0.01,100,100);
CV_MSE=[];CV_MSE_LASSO=[];
for aa = 1:k







    % Para cada lambda, ajustamos y evaluamos los modelos
    for bb=1:length(lambda_grid) 
        % RIDGE REGRESSION
        
        
        % LASSO
        
        
    end
    
end



subplot(211);plot(lambda_grid,mean(CV_MSE,1));title('RIDGE');
hold on;plot(lambda_grid(pos),val,'ro');hold off;
subplot(212);plot(lambda_grid_LASSO,mean(CV_MSE_LASSO,1));title('LASSO');
hold on;plot(lambda_grid_LASSO(pos2),val2,'ro');hold off;
pause;close;



% Para RIDGE






% Para LASSSO





fprintf('\n Ridge regression, CV lambda = %.2f, el error de test = %4.0f \n',lambda_RIDGE,MSE_test_RIDGE);
fprintf('\n LASSO, CV lambda = %.2f, el error de test = %4.0f \n',lambda_LASSO,MSE_test_LASSO);

% Reajustamos el modelo usando todas las observaciones de las que disponemos 
% y el valor de lambda seleccionado


