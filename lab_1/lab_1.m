%% <1>

fprintf('PART 1:')

x = [-4.1 -1.7 -3.7 -4 -0.1 2.1;
     -2.4 1.7 2.2 1.5 2.7 4];
t = [1 1 0 0 1 1];

net = perceptron('hardlim', 'learnp');

display(net);
net = configure(net, x, t);
view(net);

net.inputweights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = init(net);
w = net.iw{1,1}
b = net.b{1}

for i = 1 : 10
    for j = 1 : 6
        x_j = x(:, j);
        t_j = t(j);
        y = net(x_j);
        err = t_j - y;
        w = net.iw{1, 1};
        b = net.b{1};
        new_w = w + err * x_j';
        b_new = b(1) + err;
        net.iw{1, 1} = new_w;
        net.b{1} = b_new;
    end
    fprintf('w = [%f %f],  b = %f\n\n', net.iw{1, 1}, net.b{1});
end

y = net(x);
fprintf('error: %f\n\n', mae(t - y));

plotpv(x, t)
plotpc(net.iw{1, 1}, net.b{1})
grid

%%

net.inputweights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = init(net);

w = net.iw{1, 1}
b = net.b{1}

net.trainParam.epochs = 50;
net = train(net, x, t);

y = net(x);
fprintf('error: %f\n', mae(t - y))

plotpv(x, t)
plotpc(net.iw{1, 1}, net.b{1})
grid

l = -2;
r = 3;
nx = (r - l).*rand(2, 3) + l
nt = net(nx)

point = findobj(gca, 'type', 'point');
set(point, 'Color', 'red');
hold on
plotpv(nx, nt)
grid on
hold off

%% < 2 >

fprintf('PART 2:')

x = [-3.9 -4.3 4.5 0.8 2.5 0 3.9 5;
     -0.1 -0.4 -1.6 -2.8 -2.5 1.9 4.5 5];
t = [0 1 1 0 0 0 1 0];

net.inputweights{1, 1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = configure(net, x, t);
w = net.iw{1, 1}
b = net.b{1}

net.trainParam.epochs = 50;
net = train(net, x, t);

plotpv(x, t);
plotpc(net.iw{1, 1}, net.b{1});
grid;

%% < 3 >

fprintf('PART 3:')

x = [2 -2.3 -4.1 1.9 4.5 -0.7 2.6 -3.2;
     -4.7 -4.6 3.2 -1.9 -4.7 -1.2 2.9 -0.2];
t = [1 1 0 1 1 1 0 1;
     0 1 1 0 0 0 0 1];

net.inputweights{1, 1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = configure(net, x, t);
w = net.iw{1, 1}
b = net.b{1}

net.trainParam.epochs = 50;
net = train(net, x, t);

plotpv(x, t);
plotpc(net.iw{1, 1}, net.b{1});
grid;

%%

rand_points = rands(2, 5)
rand_results = net(rand_points);
plotpv(rand_points, rand_results);
plotpc(net.iw{1,1}, net.b{1});
grid;