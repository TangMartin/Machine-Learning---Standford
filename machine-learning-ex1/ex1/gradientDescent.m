function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
columns = size(X, 2)
temp = zeros(columns,1)
buffer=1:columns;
thetadefault = theta

for iter = 1:num_iters
    i = 1:m;

    temp1  = theta(1) -(alpha/m)*sum(((theta(1) + theta(2).*X(i,2)) - y(i)));
    temp2 = theta(2) -(alpha/m)*sum(((theta(1) + theta(2).*X(i,2)) - y(i)).*X(i,2));
    
    temp1  = theta(1) -(alpha/m)*sum((X*theta - y(i)));
    temp2 = theta(2) -(alpha/m)*sum((X*theta - y).*X(i,2));
    
    theta(1) = temp1
    theta(2) = temp2
    #theta = theta - (alpha/m)*(sum((X*theta - y).*(X(:,buffer))))'
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
