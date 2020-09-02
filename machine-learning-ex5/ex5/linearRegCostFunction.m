function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

#REGULARIZED COST FUCNTION
#Hypothesis
h = X * theta; #12x2 * 2x1
#Regularization
reg = (lambda/(2 * m)) * sum(theta(2:end).^2);
#Cost Function 
J = ((1 / (2 * m)) * (sum((h - y).^2))) + reg;
#G
thetazero = theta;
thetazero(1) = 0;
g = (lambda/m) * thetazero';
#REGULARIZED LINEAR REGRESSION GRADIENT
grad = (1 / m) * ((h - y)' * X) + g;;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
