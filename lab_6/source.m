clear;
clc;
%% PART 1
% Использовать слой Кохонена для выполнения кластеризации множества точек.
% Проверит качество разбиения.

%% Формируем множество случайных точек из множества.
X = [0 1.5; 0 1.5];
clusters = 8;
points = 10;
deviation = 0.1;
P = nngenc(X, clusters, points, deviation);

figure;
hold on;
grid on;
scatter(P(1, :), P(2, :), 5, [0 1 0], 'filled');

%% Создаем сеть
net = competlayer(8);
net = configure(net, P);

%% Проводим обучение
net.divideFcn = '';
net.trainParam.epochs = 50;
% Kohonen learning rule:
% a ~= 0 => dw = lr*(p'-w)
% a = 0 => dw = 0
net = train(net, P);

%% Отображаем структуру сети
view(net);

%% 5 случайных точек подаем в сеть
R = zeros(2, 5) + 1.5 * rand(2, 5);
res = vec2ind(sim(net, R))

figure;
hold on;
grid on;
scatter(P(1, :), P(2, :), 5, [0 1 0], 'filled');
scatter(net.IW{1}(:, 1), net.IW{1}(:, 2), 5, [0 0 1], 'filled');
scatter(R(1, :), R(2, :), 5, [1 0 0], 'filled');

%% PART 2:
% Использовать карту Кохонена для выполнения кластеризации множества точек

%% Сформировываем множество случайных точек
X = [0 1.5;
     0 1.5];
clusters = 8;
points = 10;
deviation = 0.1;
P = nngenc(X, clusters, points, deviation);

%% Создаем сеть
net = newsom(X, [2 4]);
net = configure(net, P);
%net = selforgmap([2 4]);

%% Производим обучение и отоброжаем графики SOM Sample Hit и SOM Weight Positins
net.divideFcn = '';
net.trainParam.epochs = 150;
net = train(net, P);

plotsomhits(net,P)
%%
plotsompos(net,P)

%%
R = zeros(2, 5) + 1.5 * rand(2, 5);
res = vec2ind(sim(net, R))

figure;
hold on;
grid on;
scatter(P(1, :), P(2, :), 5, [0 1 0], 'filled');
scatter(net.IW{1}(:, 1), net.IW{1}(:, 2), 5, [0 0 1], 'filled');
scatter(R(1, :), R(2, :), 5, [1 0 0], 'filled');
plotsom(net.IW{1, 1}, net.layers{1}.distances);

%% PART 3:
% Использовать карту Кохонена для нахождения одного из решений задачи
% Коммивояжера.

%% Генерируем и отображаем случайный набор точек
N = 20;
T = -1.5 * ones(2, N) + 3 * rand(2, N);

figure;
hold on;
grid on;
plot(T(1,:), T(2,:), '-V', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g', 'MarkerSize', 7);

%% Создаем сеть
net = newsom(T, N);
net = configure(net, T);
view(net);

%% Производим обучение
net.divideFcn = '';
net.trainParam.epochs = 600;
net = train(net, T);

%% Отображаем координаты городов
figure;
hold on;
grid on;
plotsom(net.IW{1,1}, net.layers{1}.distances);
plot(T(1,:), T(2,:), 'V', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g', 'MarkerSize', 7);

%% PART 4:
% Использовать сеть векторного квантования, обучаемую с учителем,
% (LVQ-сеть) для классификации точек в случае, когда классы не являются
% линейно разделимыми

%% 
P = [1.4 1.4 0 -0.3 -1.3 -1.4 0.9 0.3 1.1 -1 1.1 0
     0.2 0.1 -0.6 -0.1 1.1 -0.2 -0.4 0.9 1.2 -0.8 0.2 0.3];
T = [1 1 1 -1 1 1 1 -1 1 1 1 -1];

%% Отображаем входное множество и эталоны по классам
plotpv(P, max(0, T));

%% Строим вектор индексов классов
Ti = T;
Ti(Ti == 1) = 2;
Ti(Ti == -1) = 1;
Ti = ind2vec(Ti);

%% Создаем сеть
%percentage = [nnz(T(T == -1)) nnz(T(T == 1))] / numel(T);
%net = newlvq(12, percentage, 0.1);
net = lvqnet(12, 0.1);
net = configure(net, P, Ti);

%% Отображаем структуру сети
view(net)
net.IW{1,1}
net.LW{2,1}

%% Производим обучение
net.divideFcn = '';
net.trainParam.epochs = 300;
net = train(net, P, Ti);

%% Классифицируем точки
[X,Y] = meshgrid([-1.5 : 0.1 : 1.5], [-1.5 : 0.1 : 1.5]);
res = sim(net, [X(:)'; Y(:)']);
res = vec2ind(res) - 1;

figure;
plotpv([X(:)'; Y(:)'], res);
point = findobj(gca, 'type', 'line');
set(point, 'Color', 'g');
hold on;
plotpv(P, max(0, T));
hold off;
