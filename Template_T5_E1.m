function Template_T5_E1
% Este script contiene la resolución del ejercicio aplicado 1 del Tema 5
% de la asignatura 'Técnicas de Inteligencia Artificial'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Este ejercicio está relacionado con el uso de la base de datos College. 
% Trataremos de predecir el número de solicitudes de mátriculación (Apps)
% recibidas usando las otras variables de la base de datos.

% Cargamos base de datos


disp('%%%%%%%%%%%%%%%%% EJERCICIO 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')



disp('%%%%%%%%%%%%%%%%% Apartado 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 1 - Divide los datos de manera aleatoria en conjuntos de 
% entrenamiento (50%) y de test (50%). Fijar la semilla para la 
% generación de números pseudo-aleatorios a rng(13).

rng(1); % Fijamos semilla para el generado de números aleatorios



fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 2 - Ajustar un modelo de regresión lineal por mínimos cuadráticos  
% usando el conjunto de entrenamiento. Reportar el error de test obtenido.




fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 3 - Ajustar un modelo ridge regression usando el conjunto de 
% entrenamiento y con una lambda seleccionada mediante 10-fold CV. 
% Reportar el error de test obtenido.

rng(3);



fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 4 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 4 - Ajustar un modelo lasso usando el conjunto de entrenamiento 
% y con una lambda seleccionada mediante 10-fold CV. Reportar el error de 
% test obtenido y el número de estimaciones de los coeficientes diferentes de 
% cero.



fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 5 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 5 - Ajustar un modelo PCR usando el conjunto de entrenamiento y 
% con una M seleccionada mediante 10-fold CV. Reportar el error de test 
% obtenido y el valor de $m$ obtenido a través de 10-fold CV.



fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 6 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 6 - Comentar los resultados obtenidos. ¿Cómo de precisas son 
% nuestras predicciones del número de solicitudes recibidas? ¿Hay mucha 
% diferencia entre los errores de test reportados en los apartados anteriores?


