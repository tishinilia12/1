function [ o ] = parac( t, p, x0, y0, alpha )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
x1 = t.^2/p/2; 
y1 = t;
o = [x1 * cos(alpha) - y1 * sin(alpha) + x0;
     x1 * sin(alpha) + y1 * cos(alpha) + y0];
end
