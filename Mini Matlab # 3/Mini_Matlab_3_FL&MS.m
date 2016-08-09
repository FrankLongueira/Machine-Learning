% Machine Learning with Professor Keene
% Mini Matlab 3: Classification
% By Frank Longueira & Matthew Smarsch

clc;
clear all;
close all;

%% Generative Model Classifier

load('data.mat');   % Load datasets

x = {bimodal.x circles.x spiral.x unimodal.x};    % Put fields into a cell array to access data easily
targ = {bimodal.y circles.y spiral.y unimodal.y}; % Cell array of dataset labels

N = 400;        % Number of total observations
N1 = 200;       % Total observations in Class 1
N2 = 400 - N1;  % Total observations in Class 2
Pr_C1 = N1/N;   % Probability of Class 1
Pr_C2 = N2/N;   % Probability of Class 2

class1_mean = zeros(4,2);   % Pre-allocate memory to store means given in Class 1 for each data set
class2_mean = zeros(4,2);   % Pre-allocate memory to store means given in Class 2 for each data set

S1  = cell(4,1);   % Pre-allocating memory to store covariance matrices and weight vectors
S2  = cell(4,1);
S  = cell(4,1);
w = cell(4,1);
w0 = zeros(4,1);
phix1 = -8:0.1:8;  % Vector used for ployting the decision hyperplane
s = 0.5;           % Gaussian basis function parameter

% Map Circle Data to Another 2 Dimensional Space to Make it Linearly Separable
phi_circles = exp(-0.5*((repmat([1;1],1,N)-x{2}').^2)/s)'; % Points of circles mapped to another space using Gaussian basis functions

% Map Spiral Data to Another 2 Dimensional Space to Make it Linearly Separable
a_mid = (6.95555)/(4*pi);         % Constructing a spiral that separates both spirals
t_phi = linspace(0, 4*pi,400);    % Domain of spiral parameter
x1 = (a_mid.*t_phi.*cos(t_phi))'; % x1 points of dividing spiral
x2 = (-a_mid.*t_phi.*sin(t_phi))';% x2 points of dividing spiral
mag_mid = sqrt(x1.^2 + x2.^2);    % Radius of spiral at (x1,x2)
magdata = sqrt(x{3}(:,1).^2 + x{3}(:,2).^2); % Radius of data points being mapped
z = mag_mid - magdata;            % Take difference between radius of dividing spiral and the data spirals.. This separates the data.

phi_spiral = [x{3}(:,1) z];       % Matrix of spiral points that have been mapped to a new two-dimensional space

% Map Bimodal Data to Another 2 Dimensional Space to Make it Linearly Separable
blue_mean = mean(x{1}(1:200,:));  % Mean of Class 1 Bimodal
red_mean = mean(x{1}(201:400,:)); % Mean of Class 2 Bimodal
slope = (red_mean(2)-blue_mean(2))/(red_mean(1)-blue_mean(1)); % Slope of line intersecting the data
x1 = -7:0.01:7;                             % Domain of line intersecting the data
x2 = slope*(x1-red_mean(2)) + red_mean(2);  % Range of line intersecting the data

blue1 = [-7 -3];    % Partition intervals based on spread of data
red1 =  [-3 -1.155];
blue2 = [-1.155 1.039];
red2 = [1.039 5];

line = [x1' x2'];         % (x1,x2) points of the line intersecting the data
phi_bimodal = zeros(N,2); % Allocate memory
guess = zeros(N,1);       % Allocate memory  
min_dist = @(xmin,ymin)   sum(((line-repmat([xmin ymin],length(x1),1)).^2)')'; % Set up function for finding distance of a data point and every point on the line

% We now map each bimodal data point into the disjoint paritions above by looking
% at a given point's orthogonal projection onto the line and associating the
% data point with the value of x1 for the point on the line that intersects the
% orthogonal projection from the data point

 for j = 1:N
    xbin = x{1}(j,1);           % Take in first coordinate of the first bimodal data point
    ybin = x{1}(j,2);           % Take in second coordinate of the first bimodal data point
    z = min_dist(xbin,ybin);    % Find minimum distance between point and the line
    t = find(z == min(z));      % Find the index in which this minimum distance occurs
    min_spot = x1(t);           % Find the coordinate x1 of the intersection point of (x1,x2) with the orthogonal projection
    if ((min_spot < -3) & (min_spot > -7)) || ((min_spot < 1.039) & (min_spot > -1.155)) % Project this minimum coordinate into a partition listed above
        phi_bimodal(j,:) = [0 1+(abs(min_spot)/10)];
        guess(j) = 0;
    else
        phi_bimodal(j,:) = [abs(min_spot)/10 0];
        guess(j) = 1;
    end
 end
    
% Unimodal is approximately linearly separable (i.e. no need to map the data)
phi_unimodal = x{4};

% Collect all mapped data into one cell array corresponding to each dataset
phi = {phi_bimodal phi_circles phi_spiral phi_unimodal};


% Apply Generative algorithm to find weights for each dataset classification
for i=1:4
    class1_mean(i,:) = mean(phi{i}(1:200,:));   % Mean vector of class 1
    class2_mean(i,:) = mean(phi{i}(201:400,:)); % Mean vector of class 2
    S1{i} =  cov(phi{i}(1:200,:),1);            % Covariance matrix between x1 and x2 in class 1
    S2{i} =  cov(phi{i}(201:400,:),1);          % Covariance matrix between x1 and x2 in class 2
    S{i}  = (N1/N)*S1{i} + (N2/N)*S2{i};        % Covariance matrix between coordinates x1 and x2 of all data points in dataset
    invS  = inv(S{i});                          % Find inverse covariance matrix
    w{i}  = invS*(class1_mean(i,:)' - class2_mean(i,:)');   % Calculate weight vector
    w0(i) = -(1/2)*class1_mean(i,:)*invS*class1_mean(i,:)'+ (1/2)*class2_mean(i,:)*invS*class2_mean(i,:)' + log(Pr_C1/Pr_C2); % Calculate w0
    Decision_Surface{i} =  -(w{i}(1)*phix1 + w0(i))/w{i}(2);        % Store decision surface for each classification, to be plotted
end

Percent_Correct_Bimodal_Generative = (N-sum(xor(guess,targ{1})))/N % Percentage of bimodal dataset correctly separated
Percent_Correct_Circles_Generative = 1                             % Percentage of circles dataset correctly separated (i.e. linearly separable)
Percent_Correct_Spiral_Generative = 1                              % Percentage of spiral dataset correctly separated (i.e. linearly separable)
Percent_Correct_Unimodal_Generative = (N - sum(xor(((w{4}'*x{4}'+w0(4))<0)',targ{4})))/N % Percentage of unimodal dataset correctly separated


%% Logistic Regression

% Perform logistic regression as given in our textbook

LogSigmoid = @(a) 1./(1+exp(-a)); % Create logsigmoid function for ease of use in code

for j = 1:4
    Io = [ones(N,1) phi{j}];                % Design matrix
    weights = [0.1 0.1 0.1]';               % Initialize weights
    min_error = 1;                          % Inititialize error
    while min_error > 1e-3                  % Continue optimizing using Newton Raphson until sufficient error received
        error_old = min_error;              % Store old error to avoid while loop not terminating
        yn = LogSigmoid(weights'*Io')';     % Applied the logsigmoid map to the linear sum input argument
        error = yn - targ{j};               % Error between target values and values mapped by the sigmoid
        R = diag(yn.*(1-yn));               % Diagonal matrix involving mapped data values
        z = Io*weights - inv(R)*(error);    % z vector used in updating weights
        weights = inv(Io'*R*Io)*Io'*R*z;    % Update weight vector
        min_error = sum(error.^2)/N;        % Retrieve overrall average error using the previous weights
        if min_error == error_old;          % Avoid non-terminating while loop
            break
        elseif det(R) < 0.05                % Avoid singular matrix occurs and breaking the algorithm
            break
        end
    end
    w_final{j} = weights;                   % Final optimized weights at the end of the algorithm for each dataset
    Log_Decision_Surface{j} =  -(w_final{j}(2)*phix1+w_final{j}(1))/(w_final{j}(3));    % Decision surface using the just found weights from log regression
end

Percent_Correct_Bimodal_Regression = (N-sum(xor((w_final{1}(2:3)'*phi{1}'+w_final{1}(1)>0)',targ{1})))/N    % Percentage of bimodal dataset correctly separated
Percent_Correct_Circles_Regression = 1                             % Percentage of circles dataset correctly separated (i.e. linearly separable)
Percent_Correct_Spiral_Regression = 1                              % Percentage of spiral dataset correctly separated (i.e. linearly separable)
Percent_Correct_Unimodal_Regression = (N - sum(xor(((w_final{4}(2:3)'*x{4}'+w_final{4}(1))>0)',targ{4})))/N % Percentage of unimodal dataset correctly separated
   
%% Plotting

% Plot original data
Titles = {'Bimodal Dataset' 'Circles Dataset' 'Spiral Dataset' 'Unimodal Dataset'};
XLIMS = {[-0.1 0.5] [0 1.1] [-4 4] [-5 5]};
YLIMS = {[-0.2 1.8] [0.2 1.2] [-4 4] [-5 6]};

for j = 1:4
    subplot(2,2,j);
    scatter(x{j}(1:200,1),x{j}(1:200,2),'b');
    hold on 
    scatter(x{j}(201:400,1),x{j}(201:400,2),'r');
    title(Titles{j});
    xlabel('x1');
    ylabel('x2');
    legend('Class 1','Class 2');
end

% Plot Generative Classifier Results
Titles = {'Bimodal Generative Classifier' 'Circles Generative Classifier' 'Spiral Generative Classifier' 'Unimodal Generative Classifier'};
figure
for j = 1:4
    subplot(2,2,j);
    scatter(phi{j}(1:200,1),phi{j}(1:200,2),'b');
    hold on 
    scatter(phi{j}(201:400,1),phi{j}(201:400,2),'r');
    hold on
    plot(phix1,Decision_Surface{j},'g');
    title(Titles{j});
    xlim(XLIMS{j})
    ylim(YLIMS{j})
    xlabel('phi1');
    ylabel('phi2');
    legend('Class 1','Class 2', 'Decision Surface');
end

% Plot Log Regression Classifier Results
Titles = {'Bimodal Log Regression Classifier' 'Circles Log Regression Classifier' 'Spiral Log Regression Classifier' 'Unimodal Log Regression Classifier'};
figure
for j = 1:4
    subplot(2,2,j);
    scatter(phi{j}(1:200,1),phi{j}(1:200,2),'b');
    hold on 
    scatter(phi{j}(201:400,1),phi{j}(201:400,2),'r');
    hold on
    plot(phix1,Log_Decision_Surface{j},'g');
    title(Titles{j});
    xlabel('phi1');
    ylabel('phi2');
    legend('Class 1','Class 2', 'Decision Surface');
    xlim(XLIMS{j})
    ylim(YLIMS{j})
end
