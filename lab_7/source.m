clear;
clc;

%% PART 1:
% Использовать автоассоциативную сеть с узким горлом для отображения набора
% данных, выделяя первую главную компоненту данных.

%% Генерируем обучающее множество
trange = 0 : 0.025 : 2 * pi;
x = ellipse(trange, 0.5, 0.4, 0.0, 0.3, -0.1);
xseq = con2seq(x);

plot(x(1, :), x(2, :), '-r', 'LineWidth', 2);

%% Создаем многослойную сеть прямого распространения
net = feedforwardnet(1, 'trainlm');
net.layers{1}.transferFcn = 'purelin';

net = configure(net, xseq, xseq);

%% Инициализируем весовые коэффициенты и смещение сети
net = init(net);

%% Задаем параметры обучения
net.trainParam.epochs = 10000;
net.trainParam.goal = 1.0e-5;

%% Выполняем обучение сети
net = train(net, xseq, xseq);

%% Отражаем структуру сети
view(net);

%% Рассчитываем выход сети
yseq = sim(net, xseq);

%% Отображаем обучающее множество и выход сети
y = cell2mat(yseq);
plot(x(1, :), x(2, :), '-r', y(1, :), y(2, :), '-b', 'LineWidth', 2);

%% PART 2:
% Использовать автоассоциативную сеть с узким горлом для аппроксимации
% кривой на плоскости, выделяя первую нелинейную главную компоненту данных.

%% Генерируем обучающее множество
phi = 0.00 : 0.025 : 2 * pi;
rx = @(phi) (-3* phi.^2+1).* (cos(phi));
ry = @(phi) (-3* phi.^2+1).* (sin(phi));
x = [rx(phi); ry(phi)];
xseq = con2seq(x);
plot(x(1, :), x(2, :), '-r', 'LineWidth', 2);

%% Создаем сеть прямого распространения
net = feedforwardnet([10 1 10], 'trainlm');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.transferFcn = 'purelin';
net = configure(net, xseq, xseq);

%% Инициализируем сеть
net = init(net);

%% Задаем параметры обучения
net.trainParam.epochs = 10000;
net.trainParam.goal = 1.0e-5;

%% Выполняем обучение
net = train(net, xseq, xseq);

%% Отображаем структуру сети
view(net);

%% Рассчитываем выход сети
yseq = sim(net, xseq);

%% Отображаем обучающее множество и выход сети
y = cell2mat(yseq);
plot(x(1, :), x(2, :), '-r', y(1, :), y(2, :), '-b', 'LineWidth', 2);

%% PART 3:
% Применить автоассоциативную сеть с узким горлом для аппроксимации
% пространственной кривой, выделяя старшие нелинейные главные
% компоненты данных.

%% Генерируем обучающее множество
phi = 0.00 : 0.025 : 2 * pi;
rx = @(phi) (-3* phi.^2+1).* (cos(phi));
ry = @(phi) (-3* phi.^2+1).* (sin(phi));
x = [rx(phi); ry(phi); phi];
xseq = con2seq(x);
plot3(x(1, :), x(2, :), x(3, :), '-r', 'LineWidth', 2);

%% Создаем многослойную сеть прямого распространения
net = feedforwardnet([10 2 10], 'trainlm');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.transferFcn = 'purelin';
net = configure(net, xseq, xseq);

%% Инициализируем сеть
net = init(net);

%% Задаем параметры обучения
net.trainParam.epochs = 10000;
net.trainParam.goal = 1.0e-5;

%% Выполняем обучение сети
net = train(net, xseq, xseq);

%% Отражаем структуру сети
view(net);

%% Рассчитываем выход
yseq = sim(net, xseq);

%% Отображаем обучающее множество и выход сети
y = cell2mat(yseq);
plot3(x(1, :), x(2, :), x(3, :), '-r', y(1, :), y(2, :), y(3, :), '-b', 'LineWidth', 2);
