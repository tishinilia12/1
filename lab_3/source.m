%% ВАРИАНТ 15
clear;
clc;

%% PART 1
% Строим многослойную сеть прямого распространения, которая
% классифицирует точки заданной области

%% Для каждой линии генерируем множество точек
t = 0 : 0.025 : 2 * pi;

X1 = ellipse(t, 0.4, 0.15, 0, 0, pi / 6);
P1 = X1(:, randperm(numel(t), 60));
T1 = [ones(1, 60); zeros(1, 60); zeros(1, 60)];

X2 = ellipse(t, 0.7, 0.5, 0, 0, -pi / 3);
P2 = X2(:, randperm(numel(t), 100));
T2 = [zeros(1, 100); ones(1, 100); zeros(1, 100)];

X3 = parac(t, 1, 0, -0.8, pi / 2);
P3 = X3(:, randperm(numel(t), 120));
T3 = [zeros(1, 120); zeros(1, 120); ones(1, 120)];

%% Множество точек разделяем на обучающее, контрольное и тестовое
[trainInd1,valInd1,testInd1] = dividerand(60, 0.7, 0.2, 0.1);
[trainInd2,valInd2,testInd2] = dividerand(100, 0.7, 0.2, 0.1);
[trainInd3,valInd3,testInd3] = dividerand(120, 0.7, 0.2, 0.1);

%% Отображаем исходные множества точек для каждого из классов
figure;
hold on;
plot(X1(1, :), X1(2, :), '-r', 'LineWidth', 2);
plot(P1(1, trainInd1), P1(2, trainInd1), 'or', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
plot(P1(1, valInd1), P1(2, valInd1), 'rV', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'c', 'MarkerSize', 7);
plot(P1(1, testInd1), P1(2, testInd1), 'rs', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'c', 'MarkerSize', 7);

plot(X2(1, :), X2(2, :), '-g', 'LineWidth', 2);
plot(P2(1, trainInd2), P2(2, trainInd2), 'og', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g', 'MarkerSize', 7);
plot(P2(1, valInd2), P2(2, valInd2), 'gV', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'c', 'MarkerSize', 7);
plot(P2(1, testInd2), P2(2, testInd2), 'gs', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'c', 'MarkerSize', 7);

plot(X3(1, :), X3(2, :), '-b', 'LineWidth', 2);
plot(P3(1, trainInd3), P3(2, trainInd3), 'ob', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b', 'MarkerSize', 7);
plot(P3(1, valInd3), P3(2, valInd3), 'bV', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'c', 'MarkerSize', 7);
plot(P3(1, testInd3), P3(2, testInd3), 'bs', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'c', 'MarkerSize', 7);

legend('Эллипс 1', 'Обучающее множество 1', 'Контрольное множество 1', 'Тестовое множество 1',...
       'Эллипс 2', 'Обучающее множество 2', 'Контрольное множество 2', 'Тестовое множество 2',...
       'Парабола 3', 'Обучающее множество 3', 'Контрольное множество 3', 'Тестовое множество 3');
axis([-1.2 1.2 -1.2 1.2]);
grid on;
hold off;

%% Составляем обучающую выборку
Ptrain = [P1(:, trainInd1) P2(:, trainInd2) P3(:, trainInd3)];
Ttrain = [T1(:, trainInd1) T2(:, trainInd2) T3(:, trainInd3)];

Pval = [P1(:, valInd1) P2(:, valInd2) P3(:, valInd3)];
Tval = [T1(:, valInd1) T2(:, valInd2) T3(:, valInd3)];

Ptest = [P1(:, testInd1) P2(:, testInd2) P3(:, testInd3)];
Ttest = [T1(:, testInd1) T2(:, testInd2) T3(:, testInd3)];

P = [Ptrain Pval Ptest];
T = [Ttrain Tval Ttest];

%% Создаем и конфигурируем сеть
net = feedforwardnet(1, 'trainrp');
net = configure(net, [-1.2 1.2; -1.2 1.2], [0 1; 0 1; 0 1]);
net.layers{2}.transferFcn = 'tansig';

%% Задаем функцию раздеения множества и параметры разделения
net.divideFcn = 'divideind';

trnInd = size(Ptrain, 2);
tstInd = size(Pval, 2);
proInd = size(Ptest, 2);

net.divideParam.trainInd = 1 : trnInd;
net.divideParam.valInd = trnInd + 1 : tstInd;
net.divideParam.testInd = tstInd + 1 : proInd;

%% Отражаем структуру сети
view(net);

%% Инициализируем весовые коэффициенты и смещение сети
net = init(net);

net.IW{1, 1}
net.b{1}

%% Задаем параметры обучения
net.trainParam.epochs = 1500;
net.trainParam.max_fail = 1500;
net.trainParam.goal = 1.0e-5;

%% Выполняем обучение сети
net = train(net, P, T);

%% Задаем сетку и рассчитываем выход сети для узлов сетки
[X, Y] = meshgrid(-1.2 : 0.025 : 1.2, -1.2 : 0.025 : 1.2);
out = sim(net, [X(:)'; Y(:)']);

%% Кодируем цветами принадлежность выходов сети к классам

% Округляем компоненты выходных векторов
min_out = min(min(out));
max_out = max(max(out));
rout = (out - min_out) / (max_out - min_out);
out = round(rout * 10) * 0.1;

% Удаляем повторяющиеся векторы
ctable = unique(out', 'rows');

% Заменяем вектор на номер строки из таблицы цветов
n = length(X);
cmap = zeros(n, n);
for i = 1 : size(ctable, 1)
    cmap(ismember(out', ctable(i, :), 'rows')) = i;
end

% Отображаем
image([-1.2, 1.2], [-1.2, 1.2], cmap);
colormap(ctable);

%% PART 2
% Строим двухслойную сеть прямого распространения, которая выполняет
% аппроксимацию функции.

%% Создаем сетку по функции из варианта задания
t0 = 0;
tn = 3.5;
dt = 0.01;
n = (tn - t0) / dt + 1;
func = @(t)cos(-2* t.^2 +7* t);
x = func(t0 : dt : tn);

%% Создаем и конфигурируем сеть
net = feedforwardnet(1, 'trainoss');
net = configure(net, t0 : dt : tn, x);

%% Разделяем обучающую выборку
[trainInd, valInd, testInd] = dividerand(n, 0.9, 0.1, 0.0);

%% Инициализируем сеть с помощью дефолтной функции
net = init(net);

%% Задаем параметры обучения
net.trainParam.epochs = 1500;
net.trainParam.max_fail = 600;
net.trainParam.goal = 1.0e-8;

%% Выполняем обучение сети
time = t0 : dt : tn;
net = train(net, time(trainInd), x(trainInd));
y = sim(net, time(trainInd));

net.IW{1, 1}
net.IW{2, 1}
net.b{1}
net.b{2}

%% Рассчитываем выход сети и сравниваем с эталонным
sqrt(mse(x(trainInd) - y))

figure;
hold on;
plot(time(trainInd), x(trainInd), '-b');
plot(time(trainInd), y, '-r');
grid on;

figure;
plot(time(trainInd), x(trainInd) - y);
grid on;

%% То же самое для контрольного подмножества
y = sim(net, time(valInd));

net.IW{1, 1}
net.IW{2, 1}
net.b{1}
net.b{2}

sqrt(mse(x(valInd) - y))

figure;
hold on;
plot(time(valInd), x(valInd), '-b');
plot(time(valInd), y, '-r');
grid on;

figure;
plot(time(valInd), x(valInd) - y);
grid on;

%% PART 3
% Строим двухслойную нейронную сеть прямого распространения, которая будет
% выполнять аппроксимацию функции.

%%
t0 = 0.5;
tn = 4;
dt = 0.01;
n = (tn - t0) / dt + 1;
func = @(t)cos(-2* t.^2 + 7*t);
x = func(t0 : dt : tn);

net = feedforwardnet(10, 'traingda');
net = configure(net, t0 : dt : tn, x);

[trainInd, valInd, testInd] = dividerand(n, 0.9, 0.1, 0.0);
net = init(net);

net.trainParam.epochs = 600;
net.trainParam.max_fail = 600;
net.trainParam.goal = 1.0e-8;

time = t0 : dt : tn;
net = train(net, time(trainInd), x(trainInd));
y = sim(net, time(trainInd));

sqrt(mse(x(trainInd) - y))

figure;
hold on;
plot(time(trainInd), x(trainInd), '-b');
plot(time(trainInd), y, '-r');
grid on;

figure;
plot(time(trainInd), x(trainInd) - y);
grid on;

y = sim(net, time(valInd));
sqrt(mse(x(valInd) - y))

figure;
hold on;
plot(time(valInd), x(valInd), '-b');
plot(time(valInd), y, '-r');
grid on;

figure;
plot(time(valInd), x(valInd) - y);
grid on;
