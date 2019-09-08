clear;
clc;

%% Строим обучающее множество
t0 = 1.0;
tn = 3.0;
dt = 0.01;
func = @(t)sin( (sin(t)) .* (t.^3) -10);
x = func(t0 : dt : tn);
y = func(t0 + dt : dt : tn + dt);

% Конвертируем в последовательные векторы
xseq = con2seq(x);
yseq = con2seq(y);

%% Создаем сеть
delays = 1 : 5;
lr = 0.01;
net = newlin(xseq, yseq, delays, lr);

view(net);

%% Инициализируем
net.inputWeights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = init(net);

net.IW{1, 1}
net.b{1}

%% Адаптация
% Инициализируем задержку
[Xs, Xi, ~, Ts] = preparets(net, xseq, yseq);
for i = 1 : 50
    % calculates network outputs and errors and updated network
    [net, Y, E] = adapt(net, Xs, Ts, Xi);

    perf = perform(net, Ts, Y);
end
display(sqrt(perf));

%% Отображаем эталонные значения и предсказанные сетью
figure;
hold on;
grid on;
plot(t0 + 5 * dt : dt : tn, cell2mat(Ts), '-b');
plot(t0 + 5 * dt : dt : tn, cell2mat(Y), '-r');
xlabel('t');
ylabel('y');

%% PART 2:
% Обучающее множество построено в первой часте.

%% Создаем сеть
delays = 1 : 3;
% Задаем максимальную скорость обучения
lr = maxlinlr(x, 'bias');
net = newlin(xseq, yseq, delays, lr);
net = configure(net, xseq, yseq);

view(net);

%% Инициализируем сеть случайными значениями
net.inputWeights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = init(net);

%% Задаем число эпох и предельное значение критерия обучения
net.trainParam.epochs = 60;
net.trainParam.goal = 1.0e-6;
[Xs, Xi, Ai, Ts] = preparets(net, xseq, yseq);

net.trainFcn = 'trains';
net = train(net, Xs, Ts);

%%
net.IW{1, 1}
net.b{1}
display(net)

%%
% Рассчитываем выход сети и показатель качества обучения
[Y, Pf, Af, E, perf] = sim(net, Xs, Xi, Ai, Ts);
display(sqrt(perf));

% Отображаем эталонные значения и предсказанные сетью
figure;
hold on;
grid on;
plot(t0 + 3 * dt : dt : tn, cell2mat(Ts), '-b');
plot(t0 + 3 * dt : dt : tn, cell2mat(Y), '-r');
xlabel('t');
ylabel('y');

% Отображаем ошибку обучения
figure;
hold on;
grid on;
plot(t0 + 3 * dt : dt : tn, cell2mat(E));
xlabel('t');
ylabel('error');

%% 2.7

steps = 100;
xt = func(tn - 3 * dt : dt : tn + (steps - 1) * dt);
yt = func(tn - 2 * dt : dt : tn + steps * dt);
xtseq = con2seq(xt);
ytseq = con2seq(yt);
[Xst, Xit, Ait, Tst] = preparets(net, xtseq, ytseq);
[Yt, Pft, Aft, Et, perft] = sim(net, Xst, Xit, Ait, Tst);
display(sqrt(perft))

% Отображаем исходную временную последовательность
figure;
hold on;
grid on;
plot(t0 + 3 * dt : dt : tn, cell2mat(Ts), '-b');
plot(t0 + 3 * dt : dt : tn, cell2mat(Y), '-r');

% Отображаем выход сети
plot(tn: dt : tn + (steps - 1) * dt, cell2mat(Tst), '-.b');
plot(tn: dt : tn + (steps - 1) * dt, cell2mat(Yt), '-.r');
xlabel('t');
ylabel('y');

% Отображаем ошибку обучения
figure;
hold on;
grid on;
plot(tn + dt : dt : tn + steps * dt, cell2mat(Et));
xlabel('t');
ylabel('error');

%% PART 3
% Строим обучающее множество

t0 = 1.0;
tn = 6.0;
dt = 0.025;

func1 = @(t)cos(t.^2 - 10 * t + 3);
func2 = @(t)1/5 * cos(t.^2 - 10 * t + 6);
x = func1(t0 : dt : tn);
y = func2(t0 : dt : tn);

%% Расширям входное множество

D = 4;
Q = numel(t0 : dt : tn);
p = zeros(D, Q);
for i = 1 : D
    p(i, i : Q) = x(1 : Q - i + 1);
end

%% Создаем сеть

net = newlind(p, y);
Y = net(p);
display(sqrt(mse(Y - y)));

net.IW{1, 1}
net.b{1}

%% Отображаем на графике эталонные значения и предсказаные сетью

figure;
hold on;
grid on;
plot(t0:dt:tn, y, '-b');
plot(t0:dt:tn, Y, '-r');

figure;
hold on;
grid on;
plot(t0:dt:tn, Y - y);