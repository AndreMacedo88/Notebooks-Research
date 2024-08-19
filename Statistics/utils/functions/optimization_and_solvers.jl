"""
Estimation of the parameters by OLS by forming the normal equation and solving
the inverse of XᵗX.
This is less efficient than other approaches, such as SVD, but it's the most common
approach in tutorials to calculate the coefficients of a linear regression (maybe after 
gradient descent)
"""
function solver_normal_equation(y, X)
    β = inv(transpose(X) * X) * (transpose(X) * y)
    β
end

function solver_SVD()
end

function solver_QR()
end

function gradient_descent()
end