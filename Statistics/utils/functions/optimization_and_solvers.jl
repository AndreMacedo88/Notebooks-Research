"""
File that contain ways to solve Linear Squares
See main response for detailed info: 
https://stats.stackexchange.com/questions/160179/do-we-need-gradient-descent-to-find-the-coefficients-of-a-linear-regression-mode?noredirect=1&lq=1

"""

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
Solves Moore-Penrose pseudoinverse. pinv(A) * b (here: pinv(X) * y)
"""
function solve_SVD(y, X)
    # pinv(X) * y  # This is the solution with already implemented solvers

    F = svd(X) # Here F.S contains a vector with the diagonal elements of the tipical ∑ matrix of the SVD decomposition
    F.V * inv(Diagonal(F.S)) * F.U' * y # In contrast to the normal equation inverse, here a diagonal matrix is always invertible (if all the entries on its main diagonal are non-zero)
end

r"""
Similar to using the function ´qr´ and solving it (´qr(X, Val(true)) \ y´)
The basic ´qr() \ y´ a solves the linear system ´Xb = Qᵗy´ but assumes that X has full rank. 
In the algorithm for the pivoted QR, which factorizes QR = XP (where P is a permutation 
vector), there is an explicit check on the rank of X and then it computes a least square 
solution for the system.
See for more information:
    https://discourse.julialang.org/t/qr-decomposition-with-julia-how-to/92508/9
    https://de.mathworks.com/help/dsp/ref/qrsolver.html?searchHighlight=qr%20solver&s_tid=srchtitle_support_results_1_qr%20solver
"""
function solve_QR(y, X)
    # qr(X, Val(true)) \ y  # This is the solution with already implemented solvers

    F = qr(X) # Here we decompose the matrix X into QR, where R is a triangular matrix. For simplicity, here we do not perform the pivoted QR factorization version

    # The next lines solve by hand the equation QRb = y, where b are the coefs
    # inv(F.R) * transpose(F.Q) * y  # Should be avoided because of the inverse
    F.R \ (F.Q'*y)[1:size(X, 2)] # Here we solve the triangular system
end

"""
"""
function solve_gradient_descent(y, X)
end