clear;clc;close all;
flow = flow_read('000008_10.png');
u = flow(:,:,1);
v = flow(:,:,2);
mask = flow(:,:,3);
I = flow_to_color(flow);
imshow(I,[]);
[m,n] = size(u);
% [x,y] = meshgrid(1:1:n,1:1:m);
% 首先将u,v groundtruth转换成vzratio

%{
    vzratio 描述的是点在极线上面的运动，他设置了最大搜索距离，然后划分为256个等级，
    uv则直接表示出来了位移的大小，以及方向。

    现在想要让两者进行转换，其实没有必要考虑方向的问题，因为极线几何限制住了方向。
    所以现在不知道的就是这个位移的大小。

1、首先要确定vzratio max。
2、uv直接平方开根号就是距离。
3、vzratio是0-255的一个值。

%}
vMax = 0.3;
x_e = 589.7285194456671;
y_e = 179.3629421896778;
dist = sqrt(u.^2+v.^2);
f =  707.091200;

%% para estamition

[x,y]= meshgrid(1:1:n,1:1:m);

F = [2.766591891692473e-06, 0.0008746049958618121, -0.1592707696334975;
  -0.0008752177392837475, 8.723914906124037e-07, 0.5342717475818763;
  0.1558942829651084, -0.5338871161099863, 1];

l = F * [x(:),y(:),ones(m*n,1)]';
l = l';

% wx = 0.00132313138262696;
% wy = 0.002757586796563841;
% wz = 0.00135210972679234;
% uwX = (f * wy * ones(m*n,1) - wz * y(:) + wy * x(:) .* x(:) / f - wx * x(:) .* y(:) / f);
% uwY = (-f * wx * ones(m*n,1) + wz * x(:) + wy * x(:) .* y(:) / f - wx * y(:) .* y(:) / f);
% uwX = reshape(uwX,m,n);
% uwY = reshape(uwY,m,n);
x_ = x - ones(size(x))*601.887300;
x_ = x_(:);
y_ = y - ones(size(y))*183.110400;
y_ = y_(:);
l1 = l(:,1);
l2 = l(:,2);
l3 = l(:,3);
left = [ l1, l2,...
    x_.*l2-y_.*l1,...
    l1.*(x_.^2) + l2.*x_.*y_,...
    l1.*x_.*y_ + l2.*(y_.^2)];
right = -(l1.*x(:)+l2.*y(:)+l3);
para = left\right;

%%
a1 = para(1);a2 = para(2);
a3 = para(3);a4 = para(4); a5 = para(5);

tx = 0.000201632
ty = -0.000533836
tz = 0.00207736;

wp_map = zeros(m,n);
r_map = zeros(m,n);
xw_map = zeros(m,n);
yw_map = zeros(m,n);
for y = 1:1:m
    for x = 1:1:n
        d = dist(y,x);
        x_ = x - 601.887300;
        y_ = y - 183.110400;
%         xw = (a1 - a3 * y_ + a4 * x_ * x_  - a5 * x_ * y_ );
%         yw = (a2 + a3 * x_ + a4 * x_ * y_  - a5 * y_ * y_ );
xw = f*ty - tz*y_ + ty/f*x_*x_ -tz/f*x_*y_;
yw = -f*tz + tz*x_ + ty/f*x_*y_ - tx/f*y_*y_;
        r_ = sqrt((x + xw - x_e)^2+(y + yw - y_e)^2);
        wp_map(y,x) = (256*d)/((r_ + d)*vMax);
        r_map(y,x) = r_;
        xw_map(y,x) = xw;
        yw_map(y,x) = yw;
    end
end
wp = wp_map.*mask;
% wp(wp>256) = 0;
imshow(wp,[]);