clear;
clc;
%% PART 1:
% ��������� � ������� ���� ������, ������� ����� ��������� �������������
% ������������� ������. ��������� �������� ��������.

%%
% ��������� ������� ���������
% �������������� ��������� ���������.
k1 = 0 : 0.025 : 1;
p1 = sin(4 * pi * k1);
t1 = -ones(size(p1));
k2 = 2.9 : 0.025 : 4.55;
g = @(k)cos(-2*k.^2 + 7*k);
p2 = g(k2);
t2 = ones(size(p2));

R = {0; 4; 2};
P = [repmat(p1, 1, R{1}), p2, repmat(p1, 1, R{2}), p2, repmat(p1, 1, R{3}), p2];
T = [repmat(t1, 1, R{1}), t2, repmat(t1, 1, R{2}), t2, repmat(t1, 1, R{3}), t2];

Pseq = con2seq(P);
Tseq = con2seq(T);

%% ������� ����
% �������� ����������� ������� �������
net = layrecnet(1 : 2, 8, 'trainoss');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net = configure(net, Pseq, Tseq);

%% �������������� ������� ����� ��� ������� ��������
[p, Pi, Ai, t] = preparets(net, Pseq, Tseq);

%% ������ ��������� ��������
net.trainParam.epochs = 1000;
net.trainParam.goal = 1.0e-5;

%% ���������� �������� ����
net = train(net, p, t, Pi, Ai);

%% �������� ��������� ����
view(net);

%% ������������ ����� ���� ��� ���������� ������������
Y = sim(net, p, Pi, Ai);

% �������� �� ������� ��������� �������� � ������������� �����.
figure;
hold on;
plot(cell2mat(t), '-b');
plot(cell2mat(Y), '-r');
legend('���������', '�������������');

%% ��������������� ��������
Yc = zeros(1, numel(Y));
for i = 1 : numel(Y)
    if Y{i} >= 0
        Yc(i) = 1;
    else
        Yc(i) = -1;
    end
end

% ���������� ��������� ������������������ �����
display(nnz(Yc == T(3 : end)))

%% �������� ���� �� �������� � R � ������������ ����� ����.
R = {0; 3; 2};
P = [repmat(p1, 1, R{1}), p2, repmat(p1, 1, R{2}), p2, repmat(p1, 1, R{3}), p2];
T = [repmat(t1, 1, R{1}), t2, repmat(t1, 1, R{2}), t2, repmat(t1, 1, R{3}), t2];

Pseq = con2seq(P);
Tseq = con2seq(T);

[p, Pi, Ai, t] = preparets(net, Pseq, Tseq);

%% ������������ ����� �� ��������� ������������.
Y = sim(net, p, Pi, Ai);

% �������� �� ������� ��������� �������� � ������������� �����.
figure;
hold on;
plot(cell2mat(t), '-b');
plot(cell2mat(Y), '-r');
legend('���������', '�������������');

%% ��������������� ��������
Yc = zeros(1, numel(Y));
for i = 1 : numel(Y)
    if Y{i} >= 0
        Yc(i) = 1;
    else
        Yc(i) = -1;
    end
end

% ���������� ��������� ������������������ �����
display(nnz(Yc == T(3 : end)))

%% PART 2
% ��������� ���� ��������, ������� ����� ������� ������ �� ���������
% ������. ��������� ������ � ������������ ��������.

%%
p0 = [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
      -1 -1 -1 +1 +1 +1 +1 -1 -1 -1;
      -1 -1 +1 +1 +1 +1 +1 +1 -1 -1;
      -1 +1 +1 +1 -1 -1 +1 +1 +1 -1;
      -1 +1 +1 +1 -1 -1 +1 +1 +1 -1;
      -1 +1 +1 +1 -1 -1 +1 +1 +1 -1;
      -1 +1 +1 +1 -1 -1 +1 +1 +1 -1;
      -1 +1 +1 +1 -1 -1 +1 +1 +1 -1;
      -1 +1 +1 +1 -1 -1 +1 +1 +1 -1;
      -1 -1 +1 +1 +1 +1 +1 +1 -1 -1;
      -1 -1 -1 +1 +1 +1 +1 -1 -1 -1;
      -1 -1 -1 -1 -1 -1 -1 -1 -1 -1];
p1 = [-1 -1 -1 +1 +1 +1 +1 -1 -1 -1;
      -1 -1 -1 +1 +1 +1 +1 -1 -1 -1;
      -1 -1 -1 +1 +1 +1 +1 -1 -1 -1;
      -1 -1 -1 +1 +1 +1 +1 -1 -1 -1;
      -1 -1 -1 +1 +1 +1 +1 -1 -1 -1;
      -1 -1 -1 +1 +1 +1 +1 -1 -1 -1;
      -1 -1 -1 +1 +1 +1 +1 -1 -1 -1;
      -1 -1 -1 +1 +1 +1 +1 -1 -1 -1;
      -1 -1 -1 +1 +1 +1 +1 -1 -1 -1;
      -1 -1 -1 +1 +1 +1 +1 -1 -1 -1;
      -1 -1 -1 +1 +1 +1 +1 -1 -1 -1;
      -1 -1 -1 +1 +1 +1 +1 -1 -1 -1];
p2 = [+1 +1 +1 +1 +1 +1 +1 +1 -1 -1;
      +1 +1 +1 +1 +1 +1 +1 +1 -1 -1;
      -1 -1 -1 -1 -1 -1 +1 +1 -1 -1;
      -1 -1 -1 -1 -1 -1 +1 +1 -1 -1;
      -1 -1 -1 -1 -1 -1 +1 +1 -1 -1;
      +1 +1 +1 +1 +1 +1 +1 +1 -1 -1;
      +1 +1 +1 +1 +1 +1 +1 +1 -1 -1;
      +1 +1 -1 -1 -1 -1 -1 -1 -1 -1;
      +1 +1 -1 -1 -1 -1 -1 -1 -1 -1;
      +1 +1 -1 -1 -1 -1 -1 -1 -1 -1;
      +1 +1 +1 +1 +1 +1 +1 +1 -1 -1;
      +1 +1 +1 +1 +1 +1 +1 +1 -1 -1;];


p3 = [-1 -1 +1 +1 +1 +1 +1 +1 -1 -1;
      -1 -1 +1 +1 +1 +1 +1 +1 +1 -1;
      -1 -1 -1 -1 -1 -1 -1 +1 +1 -1;
      -1 -1 -1 -1 -1 -1 -1 +1 +1 -1;
      -1 -1 -1 -1 -1 -1 -1 +1 +1 -1;
      -1 -1 -1 -1 +1 +1 +1 +1 -1 -1;
      -1 -1 -1 -1 +1 +1 +1 +1 -1 -1;
      -1 -1 -1 -1 -1 -1 -1 +1 +1 -1;
      -1 -1 -1 -1 -1 -1 -1 +1 +1 -1;
      -1 -1 -1 -1 -1 -1 -1 +1 +1 -1;
      -1 -1 +1 +1 +1 +1 +1 +1 +1 -1;
      -1 -1 +1 +1 +1 +1 +1 +1 -1 -1];


p4 = [-1 +1 +1 -1 -1 -1 -1 +1 +1 -1;
      -1 +1 +1 -1 -1 -1 -1 +1 +1 -1;
      -1 +1 +1 -1 -1 -1 -1 +1 +1 -1;
      -1 +1 +1 -1 -1 -1 -1 +1 +1 -1;
      -1 +1 +1 -1 -1 -1 -1 +1 +1 -1;
      -1 +1 +1 +1 +1 +1 +1 +1 +1 -1;
      -1 +1 +1 +1 +1 +1 +1 +1 +1 -1;
      -1 -1 -1 -1 -1 -1 -1 +1 +1 -1;
      -1 -1 -1 -1 -1 -1 -1 +1 +1 -1;
      -1 -1 -1 -1 -1 -1 -1 +1 +1 -1;
      -1 -1 -1 -1 -1 -1 -1 +1 +1 -1;
      -1 -1 -1 -1 -1 -1 -1 +1 +1 -1];
p6 = [+1 +1 +1 +1 +1 +1 -1 -1 -1 -1;
      +1 +1 +1 +1 +1 +1 -1 -1 -1 -1;
      +1 +1 -1 -1 -1 -1 -1 -1 -1 -1;
      +1 +1 -1 -1 -1 -1 -1 -1 -1 -1;
      +1 +1 +1 +1 +1 +1 -1 -1 -1 -1;
      +1 +1 +1 +1 +1 +1 -1 -1 -1 -1;
      +1 +1 -1 -1 +1 +1 -1 -1 -1 -1;
      +1 +1 -1 -1 +1 +1 -1 -1 -1 -1;
      +1 +1 -1 -1 +1 +1 -1 -1 -1 -1;
      +1 +1 -1 -1 +1 +1 -1 -1 -1 -1;
      +1 +1 +1 +1 +1 +1 -1 -1 -1 -1;
      +1 +1 +1 +1 +1 +1 -1 -1 -1 -1];
p9 = [-1 -1 -1 -1 +1 +1 +1 +1 +1 +1;
      -1 -1 -1 -1 +1 +1 +1 +1 +1 +1;
      -1 -1 -1 -1 +1 +1 -1 -1 +1 +1;
      -1 -1 -1 -1 +1 +1 -1 -1 +1 +1;
      -1 -1 -1 -1 +1 +1 -1 -1 +1 +1;
      -1 -1 -1 -1 +1 +1 -1 -1 +1 +1;
      -1 -1 -1 -1 +1 +1 +1 +1 +1 +1;
      -1 -1 -1 -1 +1 +1 +1 +1 +1 +1;
      -1 -1 -1 -1 -1 -1 -1 -1 +1 +1;
      -1 -1 -1 -1 -1 -1 -1 -1 +1 +1;
      -1 -1 -1 -1 +1 +1 +1 +1 +1 +1;
      -1 -1 -1 -1 +1 +1 +1 +1 +1 +1];

%% ������� ����
net = newhop([p4(:), p2(:), p9(:)]);
view(net);

%% ������������ ����� ���� ��� ������� ������
iterations = 600;
res = sim(net, {1 iterations}, {}, p4(:));
res = reshape(res{iterations}, 12, 10);
res(res >=0 ) = 2;
res(res < 0 ) = 1;
map = [1, 1, 1; 0, 0, 0];
image(res);
colormap(map)
axis off
axis image

%% ��������� ������ ������� �� 20% � ���������� ���.
rando = rand([12, 10]);
noise_degree = 0.2;
input = p2;
for i = 1:12
    for j = 1:10
        if rando(i, j) < noise_degree
            input(i, j) = -input(i, j);
        end
    end
end

res = reshape(input, 12, 10);
res(res >=0 ) = 2;
res(res < 0 ) = 1;
map = [1, 1, 1; 0, 0, 0];
figure('Name', 'Noised');
image(res);
colormap(map)
axis off
axis image

iterations = 600;
res = sim(net, {1 iterations}, {}, input(:));
res = reshape(res{iterations}, 12, 10);
res(res >=0 ) = 2;
res(res < 0 ) = 1;
map = [1, 1, 1; 0, 0, 0];
figure('Name', 'Recognised');
image(res);
colormap(map)
axis off
axis image

%% ��������� ������ ������� �� 30% � ���������� ���.
rando = rand([12, 10]);
noise_degree = 0.3;
input = p9;
for i = 1:12
    for j = 1:10
        if rando(i, j) < noise_degree
            input(i, j) = -input(i, j);
        end
    end
end

res = reshape(input, 12, 10);
res(res >=0 ) = 2;
res(res < 0 ) = 1;
map = [1, 1, 1; 0, 0, 0];
figure('Name', 'Noised');
image(res);
colormap(map)
axis off
axis image

iterations = 600;
res = sim(net, {1 iterations}, {}, input(:));
res = reshape(res{iterations}, 12, 10);
res(res >=0 ) = 2;
res(res < 0 ) = 1;
map = [1, 1, 1; 0, 0, 0];
figure('Name', 'Recognised');
image(res);
colormap(map)
axis off
axis image

%% Part 3
% ������������� ������� ����� ��������

%% ������ ���� ��������
p = [p0(:), p1(:), p2(:), p3(:), p4(:), p6(:), p9(:)];
Q = 7;
R = 120;
IW = [p0(:)';
      p1(:)';
      p2(:)';
      p3(:)';
      p4(:)';
      p6(:)';
      p9(:)'];
b = ones(Q, 1) * R;

a = zeros(Q, Q);
for i = 1 : Q
    a(:, i) = IW * p(:, i) + b;
end

eps = 1 / (Q - 1);
LW = eye(Q, Q) * (1 + eps) - ones(Q, Q) * eps;

%% ������� ���� ��������
net = newhop(a);
net.biasConnect(1) = 0;
net.layers{1}.transferFcn = 'poslin';

net.LW{1, 1} = LW;
%net.b{1} = b;
view(net);

%% ���������� ������ �����
iterations = 600;
input = p4(:);

a1 = IW * input + b;
res = sim(net, {1 iterations}, {}, a1);
a2 = res{iterations};
ind = a2 == max(a2);
answer = IW(ind, :)';

res = reshape(answer, 12, 10);
res(res >=0 ) = 2;
res(res < 0 ) = 1;
map = [1, 1, 1; 0, 0, 0];
image(res);
colormap(map)
axis off
axis image

%% ��������� ������ ������� �� 20% � ���������� ���.
rando = rand([12, 10]);
noise_degree = 0.2;
input = p2;

for i = 1: 12
    for j = 1: 10
        if rando(i, j) < noise_degree
            input(i, j) = -input(i, j);
        end
    end
end

res = reshape(input, 12, 10);
res(res >=0 ) = 2;
res(res < 0 ) = 1;
map = [1, 1, 1; 0, 0, 0];
figure('Name', 'Noised');
image(res);
colormap(map)
axis off
axis image

input = p2(:);
a1 = IW * input + b;
res = sim(net, {1 iterations}, {}, a1);
a2 = res{iterations};
ind = a2 == max(a2);
answer = IW(ind, :)';

res = reshape(answer, 12, 10);
res(res >=0 ) = 2;
res(res < 0 ) = 1;
map = [1, 1, 1; 0, 0, 0];
figure('Name', 'Recognised');
image(res);
colormap(map)
axis off
axis image

%% ��������� ������ ������� �� 30% � ���������� ���.
rando = rand([12, 10]);
noise_degree = 0.3;
input = p9;

for i = 1: 12
    for j = 1: 10
        if rando(i, j) < noise_degree
            input(i, j) = -input(i, j);
        end
    end
end

res = reshape(input, 12, 10);
res(res >=0 ) = 2;
res(res < 0 ) = 1;
map = [1, 1, 1; 0, 0, 0];
figure('Name', 'Noised');
image(res);
colormap(map)
axis off
axis image

input = p9(:);
a1 = IW * input + b;
res = sim(net, {1 iterations}, {}, a1);
a2 = res{iterations};
ind = a2 == max(a2);
answer = IW(ind, :)';

res = reshape(answer, 12, 10);
res(res >=0 ) = 2;
res(res < 0 ) = 1;
map = [1, 1, 1; 0, 0, 0];
figure('Name', 'Recognised');
image(res);
colormap(map)
axis off
axis image
