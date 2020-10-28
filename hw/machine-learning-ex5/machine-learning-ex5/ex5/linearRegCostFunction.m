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
grad = zeros(size(theta)); # (k+1) * 1 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

### Part 1: Compute the cost
# y: m* 1
# X: m* (k+1)
# theta: (k+1) * 1 
J_1 = (y-X*theta)' * (y-X*theta)/(2*m) ; # y-X*theta --> m * 1 
theta_r = theta(2:end);
J_2 = lambda/(2*m) * (theta_r' * theta_r);
J = J_1 + J_2;


### Part 2: Compute the gradient 

grad(1) = ((X*theta-y)' * X(:,1))/m ; # y-X*theta --> m * 1 and X(:,1) --> m * 1 

grad(2:end) = ones(size(theta_r)) .* (((X*theta-y)' * X(:,2:end))'/m )  + lambda/m .* theta_r ;
# y-X*theta --> m * 1 and X(:,2:end) --> m * k 



% =========================================================================

grad = grad(:);

end
