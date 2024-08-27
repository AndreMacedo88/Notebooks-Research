using LinearAlgebra

"""
Estimation of the parameters by OLS by forming the normal equation and solving
the inverse of XᵗX.
This is less efficient than other approaches, such as SVD, but it's the most common
approach in tutorials to calculate the coefficients of a linear regression (maybe after 
gradient descent)
"""
solve_normal_equation(y, X) = inv(transpose(X) * X) * (transpose(X) * y)

"""
Estimation of the parameters by OLS by forming the normal equation. Version for just one
coefficient.
"""
function solve_normal_equation_simple(y, x, n)
    x̄ = (1 / n) .* sum(x)
    ȳ = (1 / n) .* sum(y)

    cov = ((1 / n) .* sum(y .* x)) - ((1 / (n^2)) .* sum(y) .* sum(x))
    var_x = ((1 / n) .* sum(x .^ 2)) - (((1 / n) .* sum(x))^2)

    b1 = cov / var_x
    b0 = ȳ - (b1 * x̄)

    b0, b1
end

"""
Solves Moore-Penrose pseudoinverse? pinv(A) * b (here: pinv(X) * y)
"""
function solve_SVD(y, X)
    F = svd(X)
    pinv(F.S)
end

r"""
Similar to using the function ´qr´ and solving it (´qr(X, Val(true)) \ y´)
The basic ´qr() \ y´ a solves the linear system ´Xb = Qᵗy´ but assumes that X has full rank. 
In the algorithm for the pivoted QR, there is an explicit check on the rank of X 
and then it computes a least square solution for the system.
"""
function solve_QR(y, X)
    qr(X, Val(true)) \ y
end

"""
"""
function solve_gradient_descent()
end