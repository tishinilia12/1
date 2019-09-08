clear;
clc;

%% PART 1:
% ������������ ����������������� ���� � ����� ������ ��� ����������� ������
% ������, ������� ������ ������� ���������� ������.

%% ���������� ��������� ���������
trange = 0 : 0.025 : 2 * pi;
x = ellipse(trange, 0.5, 0.4, 0.0, 0.3, -0.1);
xseq = con2seq(x);

plot(x(1, :), x(2, :), '-r', 'LineWidth', 2);

%% ������� ������������ ���� ������� ���������������
net = feedforwardnet(1, 'trainlm');
net.layers{1}.transferFcn = 'purelin';

net = configure(net, xseq, xseq);

%% �������������� ������� ������������ � �������� ����
net = init(net);

%% ������ ��������� ��������
net.trainParam.epochs = 10000;
net.trainParam.goal = 1.0e-5;

%% ��������� �������� ����
net = train(net, xseq, xseq);

%% �������� ��������� ����
view(net);

%% ������������ ����� ����
yseq = sim(net, xseq);

%% ���������� ��������� ��������� � ����� ����
y = cell2mat(yseq);
plot(x(1, :), x(2, :), '-r', y(1, :), y(2, :), '-b', 'LineWidth', 2);

%% PART 2:
% ������������ ����������������� ���� � ����� ������ ��� �������������
% ������ �� ���������, ������� ������ ���������� ������� ���������� ������.

%% ���������� ��������� ���������
phi = 0.00 : 0.025 : 2 * pi;
rx = @(phi) (-3* phi.^2+1).* (cos(phi));
ry = @(phi) (-3* phi.^2+1).* (sin(phi));
x = [rx(phi); ry(phi)];
xseq = con2seq(x);
plot(x(1, :), x(2, :), '-r', 'LineWidth', 2);

%% ������� ���� ������� ���������������
net = feedforwardnet([10 1 10], 'trainlm');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.transferFcn = 'purelin';
net = configure(net, xseq, xseq);

%% �������������� ����
net = init(net);

%% ������ ��������� ��������
net.trainParam.epochs = 10000;
net.trainParam.goal = 1.0e-5;

%% ��������� ��������
net = train(net, xseq, xseq);

%% ���������� ��������� ����
view(net);

%% ������������ ����� ����
yseq = sim(net, xseq);

%% ���������� ��������� ��������� � ����� ����
y = cell2mat(yseq);
plot(x(1, :), x(2, :), '-r', y(1, :), y(2, :), '-b', 'LineWidth', 2);

%% PART 3:
% ��������� ����������������� ���� � ����� ������ ��� �������������
% ���������������� ������, ������� ������� ���������� �������
% ���������� ������.

%% ���������� ��������� ���������
phi = 0.00 : 0.025 : 2 * pi;
rx = @(phi) (-3* phi.^2+1).* (cos(phi));
ry = @(phi) (-3* phi.^2+1).* (sin(phi));
x = [rx(phi); ry(phi); phi];
xseq = con2seq(x);
plot3(x(1, :), x(2, :), x(3, :), '-r', 'LineWidth', 2);

%% ������� ������������ ���� ������� ���������������
net = feedforwardnet([10 2 10], 'trainlm');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.transferFcn = 'purelin';
net = configure(net, xseq, xseq);

%% �������������� ����
net = init(net);

%% ������ ��������� ��������
net.trainParam.epochs = 10000;
net.trainParam.goal = 1.0e-5;

%% ��������� �������� ����
net = train(net, xseq, xseq);

%% �������� ��������� ����
view(net);

%% ������������ �����
yseq = sim(net, xseq);

%% ���������� ��������� ��������� � ����� ����
y = cell2mat(yseq);
plot3(x(1, :), x(2, :), x(3, :), '-r', y(1, :), y(2, :), y(3, :), '-b', 'LineWidth', 2);
