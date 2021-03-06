% Machine Learning - Mini Project #2
% By Matthew Smarsch & Frank Longueira

close all;
clear all;
clc;
format shortEng;

% Allocating Memory and Fixing Parameters

obs = 20;                       % Number of observations

ao = -0.3;                      % Weight ao of the linear function
a1 = 0.5;                       % Weight a1 of the linear function
alpha = 2;                      % Precision of the weights
mn_0 = [0 0];                   % Mean of prior on the weights
Sn_0 = (alpha)*eye(2);          % Covariance matrix of prior on the weights
mn = zeros(obs,2);              % Allocating memory for mean updating
Sn1 = cell(obs,1);              % Allocating memory for covariance matrix updating
Io = zeros(obs,2);              % Allocating memory for basis function matrix      

mu = 0;                         % Mean of noise
sigma = 0.2;                    % Standard deviation of noise
var = sigma^2;                  % Variance of the noise
beta = 1/var;                   % Precision of noise

          
% Generating data points
xn = -1 + (2).*rand(obs,1);                   % Generating random input values on the range [-1 1]
det_func = (ao + a1*xn);                      % Deterministic function we are trying to estimate, evaluated at the points xn
targ = det_func + normrnd(mu,sigma,obs,1);    % Add noise to deterministic function, these are the target values


% Make function for likelihood function per observation
syms f(x,t,w0,w1)      
f(x,t,w0,w1) =  ((1/(sigma*sqrt(2*pi)))*exp((-(t-(w0+w1*x))^2)/(2*var)));

Like_n = cell(obs,1);                         % Allocating memory for each likelihood function


% Generating and storing new mean vector, covariance matrix, and likelihood function for the weights, per observation
for i = 1:obs
    Io(i,:) = [1 xn(i)];
    Sn1{i} = alpha*eye(2) + beta*Io(1:i,:)'*Io(1:i,:);
    mn(i,:) = beta*inv(Sn1{i})*Io(1:i,:)'*targ(1:i,1);
    
    Like_n{i} = f(xn(i),targ(i),w0,w1); 
end

% Create meshgrid for the space of the weights, ao and a1 (w0 and w1)
range = -1:0.1:1;                               % Range space for parameters
[W00 W11] = meshgrid(range,range);

% Evaluate likelihood function of weight space for observation 1, 2, and 20
Z_Like_1 = eval(subs(Like_n{1},{w0 w1},{W00, W11}));
Z_Like_2 = eval(subs(Like_n{2},{w0 w1},{W00, W11}));
Z_Like_20 = eval(subs(Like_n{obs},{w0 w1},{W00, W11}));


% Create prior distribution of the weights and draw values from it
Prior = mvnpdf([W00(:) W11(:)],mn_0,Sn_0);
Draw_prior = mvnrnd(mn_0,Sn_0,6);
y1 = zeros(6,length(range));
for j = 1:6                                     % Generate 6 possible estimates on the deterministic function using the prior on the weights
    y1(j,:)= Draw_prior(j,1)+ Draw_prior(j,2)*range;
end

% Create posterior distribution of the weights, after 1st observation, and draw values from it
Posterior_1 = mvnpdf([W00(:) W11(:)],mn(1,:),inv(Sn1{1}));
Draw_Posterior_1 = mvnrnd(mn(1,:),inv(Sn1{1}),6);
y2 = zeros(6,length(range));                    % Generate 6 possible estimates on the deterministic function using the posterior on the weights
for j = 1:6
    y2(j,:)= Draw_Posterior_1(j,1)+ Draw_Posterior_1(j,2)*range;
end

% Create posterior distribution of the weights, after 2nd observation, and draw values from it
Posterior_2 = mvnpdf([W00(:) W11(:)],mn(2,:),inv(Sn1{2}));
Draw_Posterior_2 = mvnrnd(mn(2,:),inv(Sn1{2}),6);
y3 = zeros(6,length(range));
for j = 1:6                                     % Generate 6 possible estimates on the deterministic function using the new posterior on the weights
    y3(j,:)= Draw_Posterior_2(j,1)+ Draw_Posterior_2(j,2)*range;
end

% Create posterior distribution of the weights, after 20th observation, and draw values from it
Posterior_20 = mvnpdf([W00(:) W11(:)],mn(20,:),inv(Sn1{20}));
Draw_Posterior_20 = mvnrnd(mn(20,:),inv(Sn1{20}),6);
y4 = zeros(6,length(range));                    % Generate 6 possible estimates on the deterministic function using the final posterior on the weights
for j = 1:6
    y4(j,:)= Draw_Posterior_20(j,1)+ Draw_Posterior_20(j,2)*range;
end

% Generating Predictive Distributions for this regression after 1st, 2nd, and 20th observations

syms x

Io_x = transpose([1 x]);                                        % Basis function vector

var_pred_1 = (1/beta) + (transpose(Io_x)*inv(Sn1{1})*Io_x);     % 1st Observation
mu_pred_1 = mn(1,:)*Io_x;
Predictive_Dist_1 = eval(subs(mu_pred_1,{x},{range}));
error1 = sqrt(eval(subs(var_pred_1,{x},{range})));

var_pred_2 = (1/beta) + (transpose(Io_x)*inv(Sn1{2})*Io_x);     % 2nd Observation
mu_pred_2 = mn(2,:)*Io_x;
Predictive_Dist_2 = eval(subs(mu_pred_2,{x},{range}));
error2 = sqrt(eval(subs(var_pred_2,{x},{range})));

var_pred_20 = (1/beta) + (transpose(Io_x)*inv(Sn1{obs})*Io_x);  % 20th Observation
mu_pred_20 = mn(obs,:)*Io_x;
Predictive_Dist_20 = eval(subs(mu_pred_20,{x},{range}));
error20 = sqrt(eval(subs(var_pred_20,{x},{range})));




% Plot Figure 3.7!

subplot(4,3,2)
contourf(W00, W11, reshape(Prior,length(range),length(range)));
xlabel('\omegao')
ylabel('\omega1')
title('Prior/Posterior')
hold on
scatter(ao,a1,'+')

subplot(4,3,3)
plot(range,y1(1,:),'r',range,y1(2,:),'r',range,y1(3,:),'r',range,y1(4,:),'r',range,y1(5,:),'r',range,y1(6,:),'r');
xlabel('x')
ylabel('y')
xlim([-1 1]);
ylim([-1 1]);
title('Data Space')


subplot(4,3,4)
contourf(W00, W11, Z_Like_1);
xlabel('\omegao')
ylabel('\omega1')
title('Likelihood');
hold on
scatter(ao,a1,'+')

subplot(4,3,5)
contourf(W00, W11, reshape(Posterior_1,length(range),length(range)));
xlabel('\omegao')
ylabel('\omega1')
hold on
scatter(ao,a1,'+')

subplot(4,3,6)
plot(range,y2(1,:),'r',range,y2(2,:),'r',range,y2(3,:),'r',range,y2(4,:),'r',range,y2(5,:),'r',range,y2(6,:),'r');
xlabel('x')
ylabel('y')
hold on
scatter(xn(1),targ(1))
xlim([-1 1]);
ylim([-1 1]);


subplot(4,3,7)
contourf(W00, W11, Z_Like_2);
xlabel('\omegao')
ylabel('\omega1')
hold on
scatter(ao,a1,'+')

subplot(4,3,8)
contourf(W00, W11, reshape(Posterior_2,length(range),length(range)));
xlabel('\omegao')
ylabel('\omega1')
hold on
scatter(ao,a1,'+')

subplot(4,3,9)
plot(range,y3(1,:),'r',range,y3(2,:),'r',range,y3(3,:),'r',range,y3(4,:),'r',range,y3(5,:),'r',range,y3(6,:),'r');
xlabel('x')
ylabel('y')
xlim([-1 1]);
ylim([-1 1]);
hold on
scatter(xn(1:2),targ(1:2));


subplot(4,3,10)
contourf(W00, W11, Z_Like_20);
xlabel('\omegao')
ylabel('\omega1')
hold on
scatter(ao,a1,'+')

subplot(4,3,11)
contourf(W00, W11, reshape(Posterior_20,length(range),length(range)));
xlabel('\omegao')
ylabel('\omega1')
hold on
scatter(ao,a1,'+')

subplot(4,3,12)
plot(range,y4(1,:),'r',range,y4(2,:),'r',range,y4(3,:),'r',range,y4(4,:),'r',range,y4(5,:),'r',range,y4(6,:),'r');
xlabel('x')
ylabel('y')
xlim([-1 1]);
ylim([-1 1]);
hold on
scatter(xn(1:20),targ(1:20));

% Plot Figure 3.8 (corresponding to this specific linear regression)!

figure
det_func = ao + a1*range;
subplot(3,1,1)
errorbar(range,Predictive_Dist_1,error1,'r')
hold on
plot(range,det_func,'g')
xlim([-1 1]);
ylim([-1.5 1.5]);
scatter(xn(1),targ(1))
title('Predictive Distribution (in Red) After 1st Observation')
xlabel('x')
ylabel('t')

subplot(3,1,2)
errorbar(range,Predictive_Dist_2,error2,'r')
hold on
plot(range,det_func,'g')
xlim([-1 1]);
ylim([-1.5 1.5]);
hold on
scatter(xn(1:2),targ(1:2));
title('Predictive Distribution (in Red) After 2nd Observation')
xlabel('x')
ylabel('t')

subplot(3,1,3)
errorbar(range,Predictive_Dist_20,error20,'r')
hold on
plot(range,det_func,'g')
xlim([-1 1]);
ylim([-1.5 1.5]);
hold on
scatter(xn(1:20),targ(1:20));
title('Predictive Distribution (in Red) After 20th Observation')
xlabel('x')
ylabel('t')
