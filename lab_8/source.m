    clear;
clc;

%% PART 1:
% ������������ ���� ������� ��������������� � ������������� ���
% ������������ �������� ���������� ���� � ���������� ������������� ��������

%% ��������� ������
filename = 'sun_month_mean_activity.txt';
delimiterIn = ' ';
sun_dataset = importdata(filename, delimiterIn);

% 1690 � 3227 ������ ������ � ����� ��������� ������������������ ��������
% ��������� ������� �11. 4 - ����� ������� � ������� �������.
sun_dataset = sun_dataset(1690:3227, 4);

%% ��������� �����������
x = smooth(sun_dataset, 12);

%% ������ �������� ���������� ���������� ����. �������������� ��������.
D = 5;
ntrain = 500;
nval = 100;
ntest = 50;
trainInd = 1 : ntrain;  % 1..500
valInd = ntrain + 1 : ntrain + nval;  % 501..600
testInd = ntrain + nval + 1 : ntrain + nval + ntest;  % 601..650

%% �������� ��������� ���������
trainSet = x(1: ntrain + nval + ntest)';

%% ��������������� ��������� ���������
X = con2seq(trainSet);

%% ������� ����
net = timedelaynet(1: D, 8, 'trainlm');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';

%% ��������� ���������� ���������� ��������� � ����
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

%% ������������� ���� ��� ��������� ���������
net = configure(net, X, X);

%% �������������� ������� ������������ � �������� ���� (�� ���������)
net = init(net);

%% ������ ��������� ��������
net.trainParam.epochs = 2000;
net.trainParam.max_fail = 2000;
net.trainParam.goal = 1.0e-5;

%% ��������� �������� ����
[Xs, Xi, Ai, Ts] = preparets(net, X, X); 
net = train(net, Xs, Ts);

%% ���������� ��������� ����
view(net);

%% ������������ ����� ���� � ������ ������� ��� ���������� ���������
x_train = x(trainInd)';
[Xs, Xi, Ai, Ts] = preparets(net, con2seq(x_train), con2seq(x_train));

Y = net(Xs, Xi, Ai);

figure;
hold on;
plot(x_train, '-b');
plot([cell2mat(Xi) cell2mat(Y)], '-r');
grid on;

figure;
plot(x_train - [cell2mat(Xi) cell2mat(Y)]);
grid on;

%% ������������ ����� ���� ��� ��������� ������������
x_test = x(testInd)';
[Xs, Xi, Ai, Ts] = preparets(net, con2seq(x_test), con2seq(x_test));

% ��������� ��������� ������������ ��� ������������� ��������
x_val = x(valInd)';
Xi = x_val(length(x_val) - D + 1: length(x_val));
Xi = con2seq(Xi);

Y = net(Xs, Xi, Ai);

figure;
hold on;
plot(x_test, '-b');
plot([cell2mat(Xi) cell2mat(Y)], '-r');
grid on;

figure;
plot(x_test - [cell2mat(Xi) cell2mat(Y)]);
grid on;

%% PART 2:
% ������������ ���� ������� ��������������� � �������������� �������������
% ��� ������������� ������������ �������.

%% ���������� ��������� ��������� � ������������ � �� 5
k1 = 0 : 0.025 : 1;
p1 = sin(4 * pi * k1);
t1 = -ones(size(p1));
k2 = 2.9 : 0.025 : 4.55;
g = @(k)cos(-cos(k) .* k .* k + k);
p2 = g(k2);
t2 = ones(size(p2));

R = {6; 7; 1};
P = [repmat(p1, 1, R{1}), p2, repmat(p1, 1, R{2}), p2, repmat(p1, 1, R{3}), p2];
T = [repmat(t1, 1, R{1}), t2, repmat(t1, 1, R{2}), t2, repmat(t1, 1, R{3}), t2];

Pseq = con2seq(P);
Tseq = con2seq(T);

%% ������� ����
net = distdelaynet({0: 4, 0: 4}, 10, 'trainoss');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.divideFcn = '';

%% ������������� ���� ��� ��������� ���������
net = configure(net, Pseq, Tseq);

%% �������������� ������� ����� ��� ������� �������� 
[Xs, Xi, Ai, Ts] = preparets(net, Pseq, Tseq);

%% ������ ��������� ��������
net.trainParam.epochs = 1000;
net.trainParam.goal = 1.0e-5;

%% ������� ����
net = train(net, Xs, Ts, Xi, Ai);

%% ���������� ��������� ����
view(net);

%% ������������ � ������� ����� ���� ��� ���������� ���������
Y = net(Xs, Xi, Ai);

figure;
hold on;
grid on;
plot(cell2mat(Tseq), '-b');
plot([cell2mat(Xi) cell2mat(Y)], '-r');

%% ���������� ����� ���� � ���������� ����������
Yc = zeros(1, numel(Xi) + numel(Y));
for i = 1 : numel(Xi)
    if Xi{i} >= 0
        Yc(i) = 1;
    else
        Yc(i) = -1;
    end
end
for i = numel(Xi) + 1 : numel(Y)
    if Y{i} >= 0
        Yc(i) = 1;
    else
        Yc(i) = -1;
    end
end

% ������� ��������� ������������������ �����
display(nnz(Yc == cell2mat(Tseq)) / length(Tseq) * 100)

%% �������������� ����� ��������� ���������
R = {6; 3; 1};
P = [repmat(p1, 1, R{1}), p2, repmat(p1, 1, R{2}), p2, repmat(p1, 1, R{3}), p2];
T = [repmat(t1, 1, R{1}), t2, repmat(t1, 1, R{2}), t2, repmat(t1, 1, R{3}), t2];

Pseq = con2seq(P);
Tseq = con2seq(T);

%% ������������ ����� ����, ��������������� ����� ��������
[Xs, Xi, Ai, Ts] = preparets(net, Pseq, Tseq);

Y = net(Xs, Xi, Ai);

figure;
hold on;
grid on;
plot(cell2mat(Tseq), '-b');
plot([cell2mat(Xi) cell2mat(Y)], '-r');

%% ���������� ����� ���� � ���������� ����������
Yc = zeros(1, numel(Xi) + numel(Y));
for i = 1 : numel(Xi)
    if Xi{i} >= 0
        Yc(i) = 1;
    else
        Yc(i) = -1;
    end
end
for i = numel(Xi) + 1 : numel(Y)
    if Y{i} >= 0
        Yc(i) = 1;
    else
        Yc(i) = -1;
    end
end

% ������� ��������� ������������������ �����
display(nnz(Yc == cell2mat(Tseq)) / length(Tseq) * 100)

%% PART 3:
% ������������ ���������� ����������������� ���� � �������� ������� ���
% ������������� ���������� ������������ ������� � ���������� �������������
% ��������.

%% ������ ��������� ���������
t0 = 0;
tn = 10;
dt = 0.01;
n = (tn - t0) / dt + 1;
fun = @(k)sin(k.^2-6*k-2*pi)/4.0;
fun2 = @(y, u)y ./ (1 + y.^2) + u.^3;

u = zeros(1, n);
y = zeros(1, n);
u(1) = fun(0);

for i = 2 : n
    t = t0 + (i - 1) * dt;
    y(i) = fun2(y(i - 1), u(i - 1));
    u(i) = fun(t);
end

figure
subplot(2, 1, 1)
plot(t0 : dt : tn, u, '-b'), grid
ylabel('control')

subplot(2, 1, 2)
plot(t0 : dt : tn, y, '-r'), grid
ylabel('state')

xlabel('t')

x = u;

%% ������ ������� ���������� ���������� ����. ��������� ������������.
D = 3;

ntrain = 700;
nval = 200;
ntest = 97;

trainInd = 1 : ntrain;
valInd = ntrain + 1 : ntrain + nval;
testInd = ntrain + nval + 1 : ntrain + nval + ntest;

%% ��������� ���������
%trainSet = x(1: ntrain + nval + ntest)';
%trainTarget = y(1: ntrain + nval + ntest)';

%% ��������������� ��������� ���������
%X = con2seq(trainSet);
%T = con2seq(trainTarget);

%% ������� ����
net = narxnet(1 : 3, 1 : 3, 10);
net.trainFcn = 'trainlm';
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';

%% ������ ���������� ���������� ��������� �� ������������
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

%% ������������� ���� ��� ��������� ���������
%net = configure(net, {x(1: ntrain + nval + ntest)'; y(1: ntrain + nval + ntest)'});

%% �������������� ������� ������������ � �������� ���� (�� ���������)
net = init(net);

%% ������ ��������� ��������
net.trainParam.epochs = 600;
net.trainParam.max_fail = 600;
net.trainParam.goal = 1.0e-8;

%%
[Xs, Xi, Ai, Ts] = preparets(net, con2seq(x), {}, con2seq(y));
net = train(net, Xs, Ts, Xi, Ai);

%%
view(net);

%% ������������ ����� ���� � ������ ������� ��� ���������� ���������

Y = sim(net, Xs, Xi, Ai); 

figure
subplot(3, 1, 1)
plot(t0 : dt : tn, u, '-b'),grid 
ylabel('������') 

subplot(3, 1, 2) 
plot(t0 : dt : tn, x, '-b', t0 : dt : tn, [x(1:D) cell2mat(Y)], '-r'), grid 
ylabel('����� ����') 

subplot(3, 1, 3) 
plot(t0+D*dt : dt : tn, x(D+1:end) - cell2mat(Y)), grid 
ylabel('������') 

%% ������������ ����� ���� ��� ��������� ������������

last_elem_x = x(valInd(length(valInd)-2 : length(valInd)));
last_elem_u = u(valInd(length(valInd)-2 : length(valInd)));
inp = [con2seq(u(testInd)); con2seq(x(testInd))];
delay = [con2seq(last_elem_u); con2seq(last_elem_x)];

out_for_test = sim(net, inp, delay, Ai);
figure;
hold on;
plot(x(testInd), '.-b')
plot(cell2mat(out_for_test), '-or');
grid minor;
grid on;

figure;
hold on;
errortst = cell2mat(out_for_test) - x(testInd);
plot(errortst, '-r');
legend('������ ��� ��������� ������������');
grid minor;
grid on;
