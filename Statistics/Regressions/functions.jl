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
    @printf("Covariance of x and y: %s \nVariance of x: %s", cov, var_x)
end