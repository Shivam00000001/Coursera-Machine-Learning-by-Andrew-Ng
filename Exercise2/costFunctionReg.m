function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
val=0;
for i=1:m
  val=0;
  for k=1:n
    val+=(X(i,k)*theta(k));
  endfor
  h(i)=sigmoid(val);
  J += (-y(i)*log(h(i)))-((1-y(i))*log(1-h(i)));
endfor
J=J/m;
val=0;
for i=2:n
  val+=(theta(i)*theta(i));
endfor
val=(val*lambda/(2*m));
J+=val;


val=0;
for j=1:n
    for i=1:m
      val=0;
      for k=1:n
        val+=(X(i,k)*theta(k));
      endfor
        h(i)=sigmoid(val);
       grad(j)+=((h(i)-y(i))*X(i,j));
    endfor
    grad(j)=grad(j)/m;
    if j>1
      grad(j)+=(lambda*theta(j)/m);
    endif
endfor
#grad=grad/m;

% =============================================================

end
