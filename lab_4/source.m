%% ������� 15
clear;
clc;

%% PART 1
% ��� ���� ������� ������������ ������� ������ ������ �������������.
% (������ ������������� ����)

%% ��� ������ ����� ���������� ��������� �����
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

%% ��������� ����� ��������� �� ���������, ����������� � ��������
[trainInd1,valInd1,testInd1] = dividerand(60, 0.8, 0, 0.2);
[trainInd2,valInd2,testInd2] = dividerand(100, 0.8, 0, 0.2);
[trainInd3,valInd3,testInd3] = dividerand(120, 0.8, 0, 0.2);

%% ���������� �������� ��������� ����� ��� ������� �� �������
figure;
hold on;
plot(X1(1, :), X1(2, :), '-r', 'LineWidth', 2);
plot(P1(1, trainInd1), P1(2, trainInd1), 'or', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
plot(P1(1, testInd1), P1(2, testInd1), 'rs', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'c', 'MarkerSize', 7);

plot(X2(1, :), X2(2, :), '-g', 'LineWidth', 2);
plot(P2(1, trainInd2), P2(2, trainInd2), 'og', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g', 'MarkerSize', 7);
plot(P2(1, testInd2), P2(2, testInd2), 'gs', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'c', 'MarkerSize', 7);

plot(X3(1, :), X3(2, :), '-b', 'LineWidth', 2);
plot(P3(1, trainInd3), P3(2, trainInd3), 'ob', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b', 'MarkerSize', 7);
plot(P3(1, testInd3), P3(2, testInd3), 'bs', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'c', 'MarkerSize', 7);

legend('������ 1', '��������� ��������� 1', '�������� ��������� 1',...
       '������ 2', '��������� ��������� 2', '�������� ��������� 2',...
       '������ 3', '��������� ��������� 3', '�������� ��������� 3');
axis([-1.2 1.2 -1.2 1.2]);
grid on;
hold off;

%% ���������� ��������� �������
Ptrain = [P1(:, trainInd1) P2(:, trainInd2) P3(:, trainInd3)];
Ttrain = [T1(:, trainInd1) T2(:, trainInd2) T3(:, trainInd3)];

Pval = [P1(:, valInd1) P2(:, valInd2) P3(:, valInd3)];
Tval = [T1(:, valInd1) T2(:, valInd2) T3(:, valInd3)];

Ptest = [P1(:, testInd1) P2(:, testInd2) P3(:, testInd3)];
Ttest = [T1(:, testInd1) T2(:, testInd2) T3(:, testInd3)];

P = [Ptrain Pval Ptest];
T = [Ttrain Tval Ttest];

%% ������� ������������� ����
spread = 0.3;
net = newpnn(Ptrain, Ttrain, spread);
view(net);

%% ��������� �������� �������� (��������� ���������)
Ytrain = vec2ind(sim(net, Ptrain));

% ������� ���������� ��������� ������������������ �����
display(nnz(Ytrain == vec2ind(Ttrain)))

%% ��������� �������� �������� (�������� ���������)
Ytrain = vec2ind(sim(net, Ptest));

% ������� ���������� ��������� ������������������ �����
display(nnz(Ytrain == vec2ind(Ttest)))

%% �������� ������� �������������� ������� ���� � �������
[X, Y] = meshgrid(-1.2 : 0.025 : 1.2, -1.2 : 0.025 : 1.2);
out = sim(net, [X(:)'; Y(:)']);

% ��������� ���������� �������� ��������
min_out = min(min(out));
max_out = max(max(out));
pre = (out - min_out) / (max_out - min_out);
out = round(pre * 10) * 0.1;

% ������� ������������� �������
ctable = unique(out', 'rows');

% �������� ������ �� ����� ������ �� ������� ������
n = length(X);
cmap = zeros(n, n);
for i = 1 : size(ctable, 1)
    cmap(ismember(out', ctable(i, :), 'rows')) = i; 
end

% ����������
image([-1.2, 1.2], [-1.2, 1.2], cmap);
colormap(ctable);

%% ������� ����
spread = 0.1;
net = newpnn(Ptrain, Ttrain, spread);
view(net);

%% �������� ������� �������������� ������� ���� � �������
[X, Y] = meshgrid(-1.2 : 0.025 : 1.2, 1.2 : -0.025 : -1.2);
out = sim(net, [X(:)'; Y(:)']);

% ��������� ���������� �������� ��������
min_out = min(min(out));
max_out = max(max(out));
pre = (out - min_out) / (max_out - min_out);
out = round(pre * 10) * 0.1;

% ������� ������������� �������
ctable = unique(out', 'rows');

% �������� ������ �� ����� ������ �� ������� ������
n = length(X);
cmap = zeros(n, n);
for i = 1 : size(ctable, 1)
    cmap(ismember(out', ctable(i, :), 'rows')) = i; 
end

% ����������
image([-1.2, 1.2], [-1.2, 1.2], cmap);
colormap(ctable);

%% PART 2
% ��� ���� ������� ������������ ������� ������ ������ 
% ������������� (������ ���� � ����������� ��������� ����������)

% ������ 2.1 - 2.4 ������������� ������� 1.1 - 1.4

%% ������� ����
spread = 0.3;
goal = 1.0e-5;
net = newrb(Ptrain, Ttrain, goal, spread);
view(net);

%% ��������� �������� �������� (��������� ���������)
Ytrain = vec2ind(sim(net, Ptrain));

% ������� ���������� ��������� ������������������ �����
display(nnz(Ytrain == vec2ind(Ttrain)))

%% ��������� �������� �������� (�������� ���������)
Ytrain = vec2ind(sim(net, Ptest));

% ������� ���������� ��������� ������������������ �����
display(nnz(Ytrain == vec2ind(Ttest)))

%% �������� ������� �������������� ������� ���� � �������
[X, Y] = meshgrid(-1.2 : 0.025 : 1.2, -1.2 : 0.025 : 1.2);
out = sim(net, [X(:)'; Y(:)']);

% ��������� ���������� �������� ��������
max_val = max(max(out));
min_val = min(min(out));
pre = (out - min_val) / (max_val - min_val);
out = round(pre * 10) * 0.1;

% ������� ������������� �������
ctable = unique(out', 'rows');

% �������� ������ �� ����� ������ �� ������� ������
n = length(X);
cmap = zeros(n, n);
for i = 1 : size(ctable, 1)
    cmap(ismember(out', ctable(i, :), 'rows')) = i;
end

% ����������
image([-1.2, 1.2], [-1.2, 1.2], cmap);
colormap(ctable);

%% ������� ����
spread = 0.1;
goal = 1.0e-5;
net = newrb(Ptrain, Ttrain, goal, spread);
view(net);

%% �������� ������� �������������� ������� ���� � �������
[X, Y] = meshgrid(-1.2 : 0.025 : 1.2, -1.2 : 0.025 : 1.2);
out = sim(net, [X(:)'; Y(:)']);

% ��������� ���������� �������� ��������
min_out = min(min(out));
max_out = max(max(out));
pre = (out - min_out) / (max_out - min_out);
out = round(pre * 10) * 0.1;

% ������� ������������� �������
ctable = unique(out', 'rows');

% �������� ������ �� ����� ������ �� ������� ������
n = length(X);
cmap = zeros(n, n);
for i = 1 : size(ctable, 1)
    cmap(ismember(out', ctable(i, :), 'rows')) = i; 
end

% ����������
image([-1.2, 1.2], [-1.2, 1.2], cmap);
colormap(ctable);

%% PART 3
% ������ ���������-������������� ��������� ����, �������
% ����� ��������� ������������� �������.

%% ������� ����� �� ������� �� �������� �������
t0 = 0.5;
tn = 4;
dt = 0.01;
n = (tn - t0) / dt + 1;
func = @(t)cos(-cos(t) .* t.^2 + t);
x = func(t0 : dt : tn);

%% ��������� �� ��������� � �������� ���������
[trainInd, valInd, testInd] = dividerand(n, 0.9, 0.0, 0.1);

%% ������� ����
spread = dt;
time = t0 : dt : tn;
net = newgrnn(time(trainInd), x(trainInd), spread);

%% ���������� ��������� ���� � ����������� ��������.
view(net);

net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

%% ������������ ����� ����. ���������� ��������� �������� �
% ������������� �����. ���������� ������ ��������.
y = sim(net, t0 : dt : tn);

sqrt(mse(x(trainInd) - y(trainInd)))

figure;
hold on;
plot(t0 : dt : tn, x, '-b');
plot(t0 : dt : tn, y, '-r');
grid on;

figure;
plot(t0 : dt : tn, x - y);
grid on;

%% �� �� �������� ��� �������� �������
net = newgrnn(time(testInd), x(testInd), spread);

view(net);

net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

y = sim(net, t0 : dt : tn);

sqrt(mse(x(testInd) - y(testInd)))

figure;
hold on;
plot(t0 : dt : tn, x, '-b');
plot(t0 : dt : tn, y, '-r');
grid on;

figure;
plot(t0 : dt : tn, x - y);
grid on;

%% ���������� �� ����. � ����. ��-�� � �����. 80% � 20%
[trainInd, valInd, testInd] = dividerand(n, .8, 0, .2);

%% �� ��������� �������� ���������, ������������ �������� ��������
% � ���������� �� ������� �������� ������� � �������� �����
net = newgrnn(time(trainInd), x(trainInd), spread);

y = sim(net, t0 : dt : tn);

sqrt(mse(x(trainInd) - y(trainInd)))

figure;
hold on;
plot(t0 : dt : tn, x, '-b');
plot(t0 : dt : tn, y, '-r');
grid on;

figure;
plot(t0 : dt : tn, x - y);
grid on;