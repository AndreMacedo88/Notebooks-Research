"""
Compute OLS estimators of the parameters.
"""
function OLS_estimation(x::Vector{<:Real}, y::Vector{<:Real})
    n = length(x)
    
    x̄ = (1/n) .* sum(x)
    ȳ = (1/n) .* sum(y)
    @printf("mean x: %s, mean y: %s\n", x̄, ȳ)

    cov = ((1/n) .* sum(y.*x)) - ((1/(n^2)) .* sum(y) .* sum(x))
    var_x = ((1/n) .* sum(x.^2)) - (((1/n) .* sum(x))^2)
    @printf("Covariance of x and y: %s \nVariance of x: %s\n", cov, var_x)

    β1 = cov / var_x
    β0 = ȳ - (β1 * x̄)
    @printf("β0: %s \nβ1: %s\n", β0, β1)

end