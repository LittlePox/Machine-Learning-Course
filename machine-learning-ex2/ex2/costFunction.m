function [J, grad] = costFunction(theta, X, y, print=false)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
eps = 1.4901161193847656e-08

J = ((y - 1)' * log(1 - sigmoid(X*theta) + eps) - y' * log(sigmoid(X*theta) + eps)) / m;

grad = (sigmoid(X*theta) - y)' * X / m;

if (print)
    sigmoid(X*theta)
end

% =============================================================

end
