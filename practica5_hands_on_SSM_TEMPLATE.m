function practica5_hands_on_SSM_TEMPLATE
% Este script contiene la resolución del tutorial práctico del Tema 5
% de la asignatura 'Técnicas de Inteligencia Artificial'

close all;
clc;

load Hitters.mat
% Nombre de las variables
var_names = Hitters.Properties.VariableNames;

% Dimensiones de la base de datos original
size(Hitters)

% Remover valores NaN
Hitters = rmmissing(Hitters);

% Dimensiones de la base de datos sin valores NaN
size(Hitters)

disp('%%%%%%%%%%%%%%%%%%%%% SELECCIÓN GRADUAL %%%%%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

% Binarizo las variables cualitativas League, Division y NewLeague
D = dummyvar(categorical(Hitters{:,14}));
League_A = D(:,1);

D = dummyvar(categorical(Hitters{:,15}));
Division_E = D(:,1);

D = dummyvar(categorical(Hitters{:,20}));
NewLeague_A = D(:,1);

% Selección gradual hacia adelante

X = [Hitters{:,1:13} League_A Division_E, Hitters{:,16:18}, NewLeague_A];
Y = Hitters.Salary;toc

% Egiten dituen pauso guztiak erakusteko:
tic
opt = statset('Display', 'iter');
[inmodel_fwd, history_fwd] = sequentialfs(@regLIN, X, Y, 'nfeatures', size(X,2), 'cv', 'none', 'direction','forward', 'options',opt);
toc

figure(1)
image(history_fwd.In.*1000)

% Seleccionar el mejor modelo como aquel que menor R2 ajustado presente
r2_adj_fwd=zeros(1,size(history_fwd.In,1));
aic=zeros(1,size(history_fwd.In,1));
bic=zeros(1,size(history_fwd.In,1));
for cc=1:size(history_fwd.In,1)
    Xtrain = X(:, history_fwd.In(cc,:));
    mdl = fitlm(Xtrain, Y);
    r2_adj_fwd(cc) = mdl.Rsquared.Adjusted;
    aic(cc) = mdl.ModelCriterion.AIC;
    bic(cc) = mdl.ModelCriterion.BIC;
end

[val, pos] = max(r2_adj_fwd);
[val_bic, pos_bic] = min(bic);
[val_aic, pos_aic] = min(aic);

fprintf('\n SG hacia adelante (sequentialfs) mejor modelo en base a R2 ajustado\n #predictores = %d \n R2 ajustado = %4.3f \n\n',pos,val);
fprintf('\n SG hacia adelante (sequentialfs) mejor modelo en base a AIC\n #predictores = %d \n AIC = %4.3f \n\n',pos_aic,val_aic);
fprintf('\n SG hacia adelante (sequentialfs) mejor modelo en base a BIC\n #predictores = %d \n BIC ajustado = %4.3f \n\n',pos_bic,val_bic);

figure(2)
subplot(131);plot(r2_adj_fwd,'o-');xlabel('# predictores');ylabel('R^2 ajustado');
hold on;line(pos,val,'LineStyle','none','Marker','o','MarkerEdgeColor','r',...
    'MarkerFaceColor','r','MarkerSize',8);hold off;
subplot(132);plot(aic,'o-');xlabel('# predictores');ylabel('AIC');
hold on;line(pos_aic,val_aic,'LineStyle','none','Marker','o','MarkerEdgeColor','r',...
    'MarkerFaceColor','r','MarkerSize',8);hold off;
subplot(133);plot(bic,'o-');xlabel('# predictores');ylabel('BIC');
hold on;line(pos_bic,val_bic,'LineStyle','none','Marker','o','MarkerEdgeColor','r',...
    'MarkerFaceColor','r','MarkerSize',8);hold off;
hold off;
pause;close all;


% Selección gradual hacia atrás
% Egiten dituen pauso guztiak erakusteko:
tic
opt = statset('Display', 'iter');
[inmodel_bwd, history_bwd] = sequentialfs(@regLIN, X, Y, 'nfeatures', 1, 'cv', 'none', 'direction','backward', 'options',opt)
toc


% Seleccionar el mejor modelo como aquel que menor R2 ajustado presente
r2_adj_bwd=zeros(1,size(history_bwd.In,1));
aic=zeros(1,size(history_bwd.In,1));
bic=zeros(1,size(history_bwd.In,1));
for cc=1:size(history_bwd.In,1)
    Xtrain = X(:, history_bwd.In(cc,:));
    mdl = fitlm(Xtrain, Y);
    r2_adj_bwd(cc) = mdl.Rsquared.Adjusted;
    aic(cc) = mdl.ModelCriterion.AIC;
    bic(cc) = mdl.ModelCriterion.BIC;
end

% Seleccionar el mejor modelo como aquel que menor MSE de CV presente.
r2_adj_bwd = flip(r2_adj_bwd);

[val, pos] = max(r2_adj_bwd);

% Beharbada hauek izatea interesatzen zaigulako, aurrekoaren era berean lor
% daitezke
%[val_bic, pos_bic] = min(bic);
%[val_aic, pos_aic] = min(aic);

figure(3)
image(history_fwd.In.*1000);

figure(4)
plot(r2_adj_bwd,'o-');xlabel('# predictores');ylabel('R^2 ajustado');
hold on;line(pos,val,'LineStyle','none','Marker','o','MarkerEdgeColor','r',...
    'MarkerFaceColor','r','MarkerSize',8);hold off;
pause;close all;
fprintf('\n SG hacia atrás (sequentialfs) mejor modelo en base a R2 ajustado\n #predictores = %d \n R2 ajustado = %4.3f \n\n',length(r2_adj_bwd)-pos+1,val);


% Creamos particiones
rng(13);

k = 10;
c = cvpartition(length(Y), "KFold", k);

CV_MSE=[];
tic
for aa = 1:k

    pos_train = c.training(aa);
    pos_test = c.test(aa);

    Xtrain = X(pos_train,:); 
    Xtest  = X(pos_test,:);
    Ytrain = Y(pos_train,:);
    Ytest  = Y(pos_test,:);
    % SG hacia adelante 
    opt = statset('Display', 'iter');
    % CV mantentzen dugu, guk egiten diogulako cross-validation-a eskuz!! :O!!
    [inmodel_fwd, history_fwd] = sequentialfs(@regLIN, Xtrain, Ytrain, 'nfeatures', 19, 'cv', 'none', 'direction','forward', 'options',opt);
    
    % Evaluamos los modelos seleccionados
    for cc=1:size(history_fwd.In,1)
        Xtrain_1 = Xtrain(:, history_fwd.In(cc,:));
        Xtest_1 = Xtest(:, history_fwd.In(cc,:));
        mdl = fitlm(Xtrain_1, Ytrain);
        CV_MSE(aa, cc) = mean((Ytest-predict(mdl, Xtest_1)).^2);
    end
end

[val, pos] = min(mean(CV_MSE,1));


toc

figure(5)
image(history_fwd.In.*1000);


figure(6)
plot(mean(CV_MSE,1),'o-');xlabel('# predictores');ylabel('MSE CV');
hold on;line(pos,val,'LineStyle','none','Marker','o','MarkerEdgeColor','r',...
    'MarkerFaceColor','r','MarkerSize',8);hold off;
pause;close all;

% SG hacia adelante 

[inmodel_fwd, history_fwd] = sequentialfs(@regLIN, X, Y, 'nfeatures', pos, 'cv', 'none', 'direction','forward', 'options',opt);

% Ajustamos modelo para toda la base de datos
predNames = var_names;
predNames(19) = [];
mdl_final = fitlm(X(:,inmodel_fwd),Y, 'VarNames', [predNames(inmodel_fwd), var_names{19}])


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% USER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function criterio = regLIN(Xtrain,ytrain)
    % Ajustamos modelo de regresión lineal
    mdl = fitlm(Xtrain, ytrain);    
    % Para elegir el modelo de k predictores con mayor R2
    criterio = 1-mdl.Rsquared.Ordinary;
end