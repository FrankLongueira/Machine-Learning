% Machine Learning - Mini Project #1 
% By Frank Longueira & Matthew Smarsch

close all;
clear all;
clc;

num = 1e2;                          % Number of observations
iter = 1e3;                         % Number of iterations
p = 0.45;                           % Probability of success for each Bernoulli trial (this is what we are trying to estimate for the binmoial case)
mu = 13;                            % Mean of Gaussian random variable we are trying to estimate
var = 2;                            % Variance of Gaussian random variable we are trying to estimate

binomial_MSE_ML_iter = zeros(iter,num);      % Allocate memory for MSE for each iteration
gaussian_MSE_ML_iter = zeros(iter,num);
binomial_MSE_conj_iter = zeros(iter,num);
gaussian_MSE_conj_iter = zeros(iter,num+1);
gaussian_MSE_conj_var_iter = zeros(iter,num);
for j = 1:iter
    
    a = zeros(1,num+1);                         % Allocate memory for hyperparameters "a"
    b = zeros(1,num+1);                         % Allocate memory for hyperparameter "b"
    muo = zeros(1,num+1);                       % Allocate memory for hyperparameter "muo"
    v = zeros(1,num+1);                         % Allocate memory for hyperparameter "v"
    alpha = zeros(1,num+1);                     % Allocate memory for hyperparameter "alpha"
    beta = zeros(1,num+1);                      % Allocate memory for hyperparameter "beta"
    a(1)= 1;                                    % Initialize hyperparameter "a"
    b(1)= 1;                                    % Initialize hyperparameter "b"
    muo(1) = 1;                                 % Initialize hyperparameter "muo"
    v(1) = 1;                                   % Initialize hyperparameter "v"
    alpha(1) = 1;                               % Initialize hyperparameter "alpha"
    beta(1) = 1;
    
    
    binomial_ML_estimate = zeros(1,num);         % Allocate memory to store estimates
    gaussian_ML_estimate = zeros(1,num);         
    binomial_conj_estimate = zeros(1,num);       
    var_ML_estimate = zeros(1,num);              
    rv_vector = [binornd(1,p,1,num); normrnd(mu,sqrt(var),1,num)];% Observations of random variables are obtained
    var_conj_gauss_est = zeros(1,num);
    
    for i=1:num
        binomial_ML_estimate(1,i) = sum(rv_vector(1,1:i))/i;     % ML Estimate for mean by taking sample mean of observations
        gaussian_ML_estimate(1,i) = sum(rv_vector(2,1:i))/i;     % ML Estimate for mean by taking sample mean of observations
        var_ML_estimate(1,i) = sum((rv_vector(2,1:i)-gaussian_ML_estimate(1,i)).^2)/i; %ML Estimate for variance of gaussian
        a(i+1) = a(1) + sum(rv_vector(1,1:i));                   % Update hyperparameter "a"
        b(i+1) = b(1) + (i-sum(rv_vector(1,1:i)));               % Update hyperparameter "b"
        binomial_conj_estimate(i) = (a(i+1))/(a(i+1) + b(i+1));  % Conjugate prior estimate on p
        
        muo(i+1) = (v(1)*muo(1)+sum(rv_vector(2,1:i)))/(v(1)+i); % Update hyperparamter "mu"
        v(i+1) = v(1)+i;                                         % Update hyperparameter "v"
        alpha(i+1) = alpha(1)+(i/2);                             % Update hyperparameter "alpha"
        beta(i+1) = beta(1)+(1/2)*(sum(rv_vector(2,1:i)-sum(rv_vector(2,1:i))/i))+(i*v(1))/(v(1)+i)*0.5*(sum(rv_vector(2,1:i))/i - muo(1))^2; % Update hyperparameter "beta"
        var_conj_gauss_est(i) = beta(i)/alpha(i);
        
    end
    
    binomial_MSE_ML_iter(j,:) = (binomial_ML_estimate-p).^2;       % Create binomial ML MSE  matrix for every iteration
    gaussian_MSE_ML_iter(j,:) = (gaussian_ML_estimate-mu).^2;      % Create gaussian ML MSE  matrix for every iteration
    gaussian_Var_MSE_iter(j,:) = (var_ML_estimate-var).^2;         
    binomial_MSE_conj_iter(j,:) = (binomial_conj_estimate-p).^2;   % Create binomial conjugate mean squared error  vector for every iteration
    gaussian_MSE_conj_iter(j,:) = (muo-mu).^2;                     % Create gaussian conjugate mean squared error  vector for every iteration
    gaussian_MSE_conj_var_iter(j,:) = (var_conj_gauss_est - var).^2; % Create gaussian conjugate mean squared error for variance vector for every iteration
    end
    
    % Plot MSE for each random variable estimation
    
binomial_ML_MSE = mean(binomial_MSE_ML_iter);
gaussian_ML_MSE = mean(gaussian_MSE_ML_iter);
gaussian_ML_Var = mean(gaussian_Var_MSE_iter);
binomial_Conj_MSE = mean(binomial_MSE_conj_iter);
gaussian_Conj_MSE = mean(gaussian_MSE_conj_iter);
gaussian_Conj_Var_MSE = mean(gaussian_MSE_conj_var_iter);

figure

% Plot sequential posterior distributions on Binomial parameter, p
for i=1:num+1
    x = 0:0.01:1;
    y1 = betapdf(x,a(i),b(i));
    plot(x,y1)
    title('Sequential Posterior Distributions on Binomial Paremeter, p')
    ylabel('PDF');
    xlabel('p');
    legend(['\alpha = ' int2str(a(i)) ', \beta = ' int2str(b(i))]);
    ylim([0 50])
    drawnow;
end

 % Plot sequential joint posterior distributions on the parameters mu and tau
 figure
for i=15:num
    x = 5:0.5:15;
    tau = 0:0.5:10;
    [X Tau] = meshgrid(x,tau);
    x1=(beta(i).^(alpha(i)))*(sqrt(v(i)));
    x2 = 1/(gamma(alpha(i))*sqrt(2*pi));
    x3 = (Tau.^(alpha(i)-0.5)).*exp(-beta(i).*Tau);
    x4 = exp((-v(i).*Tau.*((X-muo(i)).^2))/2);
    
    x = 5:0.5:15;
    tau = 0:0.5:10;
     
    y5 = x1.*x2.*x3.*x4;
    surf(x,tau,y5)
    title('Sequential Posterior Distributions on Gaussian Paremeters, \mu & \tau')
    zlabel('PDF')
    ylabel('\tau');
    xlabel('\mu');
    legend(['mu = ' int2str(muo(i)) ', v = ' int2str(v(i)) ', alpha = ' int2str(alpha(i)) ', beta = ' int2str(beta(i))]);
    drawnow;
end

% Plot MSE graphs for each estimation

figure
subplot(2,1,1)
plot(1:num,binomial_ML_MSE, 'r')
title('ML Estimation: MSE for Binomial Parameter, p')
ylabel('Mean Squared Error')
xlabel('Number of observations')
xlim([1 num])

subplot(2,1,2)
plot(1:num,binomial_Conj_MSE, 'b')
title('Conjugate Prior Estimation: MSE for Binomial Parameter, p')
ylabel('Mean Squared Error')
xlabel('Number of observations')
xlim([1 num])

figure
subplot(2,2,1)
plot(1:num,gaussian_ML_MSE,'r')
title('ML Estimation: MSE for Gaussian Mean,\mu')
ylabel('Mean Squared Error')
xlabel('Number of observations')
xlim([1 num])

subplot(2,2,2)
plot(1:num,gaussian_ML_Var,'r')
title('ML Estimation: MSE for Gaussian Variance,\sigma^2')
ylabel('Variance Squared Error')
xlabel('Number of observations')
xlim([1 num])

subplot(2,2,3)
plot(1:num+1,gaussian_Conj_MSE,'b')
title('Conjugate Prior Estimation: MSE for Gaussian Mean,\mu')
ylabel('Mean Squared Error')
xlabel('Number of observations')
xlim([1 num])

subplot(2,2,4)
plot(1:num,gaussian_Conj_Var_MSE,'b')
title('Conjugate Prior Estimation: MSE for Gaussian Variance,\sigma^2')
ylabel('Mean Squared Error')
xlabel('Number of observations')
xlim([1 num])

