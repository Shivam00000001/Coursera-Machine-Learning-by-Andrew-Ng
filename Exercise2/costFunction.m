function [J, grad] = costFunction(theta, X, y)
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
#z=X*theta;
#h(X)=sigmoid(z);
for i=1:m
  h(i)=sigmoid(X(i,1)*theta(1)+X(i,2)*theta(2)+X(i,3)*theta(3));
  J+= (-y(i)*log(h(i)))-((1-y(i))*log(1-h(i)));
endfor
J=J/m;
for i=1:m
  h(i)=sigmoid(X(i,1)*theta(1)+X(i,2)*theta(2)+X(i,3)*theta(3));
  grad(1)+=(h(i)-y(i))*X(i,1);
  grad(2)+=(h(i)-y(i))*X(i,2);
  grad(3)+=(h(i)-y(i))*X(i,3);
endfor
grad=grad/m;







% =============================================================

end
