residuals(coefs, y, X) = y .- (coefs' * X')

residuals_simple(intercept, coefs, y, x) = y .- (coefs .* x .+ intercept)

SSE(e) = sum(e_ -> e_^2, e) # anonymous to avoid allocating space to the squared values

MSE(SSE, df) = SSE / df

function SE_simple(residual_se, x, n, type)
    X̄ = sum(x) / n

    if type == "intercept"
        return residual_se * √((1 / n) + ((X̄ .^ 2) / sum((x .- X̄) .^ 2)))
    elseif type == "predictor"
        return residual_se / √(sum((x .- X̄) .^ 2))
    else
        @warn "The type of variable does not exist. Unable to calculate SE." type
    end
end

SST(y) = sum((y .- mean(y)) .^ 2)

R_adjusted(SSE, SST, n, df) = 1 - (((n - 1) * SSE) / (df * SST))

t_statistic_parameters(coef, SE) = coef / SE

function ttest(t, df)
    model = TDist(df)

    # return 2-sided p-value
    (1 - cdf(model, abs(t)) + cdf(model, -abs(t)))
end

function confidence_interval(b, n, SE, α=0.05)
    model = TDist(n - 2)
    t_ = quantile.(model, [α / 2, 1 - (α / 2)])  # returns a vector with both quantiles
    correction = t_ * SE

    # return a vector containing the confidence interval
    b .+ correction
end

