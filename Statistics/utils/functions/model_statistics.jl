using LinearAlgebra

calculate_residuals(coefs, y, X) = y .- (X * coefs)

calculate_residuals_simple(intercept, coefs, y, x) = y .- (coefs .* x .+ intercept)

calculate_SSE(e) = sum(e_ -> e_^2, e) # anonymous to avoid allocating space to the squared values

# calculate_SSE(e) = sum(e .^ 2)

calculate_MSE(SSE, df) = SSE / df

function calculate_SE_simple(residual_se, x, n, type)
    X̄ = sum(x) / n

    if type == "intercept"
        return residual_se * √((1 / n) + ((X̄ .^ 2) / sum((x .- X̄) .^ 2)))
    elseif type == "predictor"
        return residual_se / √(sum((x .- X̄) .^ 2))
    else
        @warn "The type of variable does not exist. Unable to calculate SE." type
    end
end

function calculate_SE_smallsamples(MSE, X, df)
    covariance_mtx = MSE * (inv(transpose(X) * X))
    variance = diag(covariance_mtx)
    sqrt.(variance)
end

calculate_SST(y) = sum((y .- mean(y)) .^ 2)

calculate_R_adjusted(SSE, SST, n, df) = 1 - (((n - 1) * SSE) / (df * SST))

calculate_t_statistic(coefs, SE) = coefs ./ SE

function perform_ttest(t, df)
    model = TDist(df)

    # return 2-sided p-value
    (1 .- cdf(model, abs.(t)) + cdf(model, -abs.(t)))
end

function calculate_confidence_interval(b, df, SE, α=0.05)
    model = TDist(df)
    t_ = quantile.(model, [α / 2, 1 - (α / 2)])  # returns a vector with both quantiles
    correction = t_ * SE

    # return a vector containing the confidence interval
    b .+ correction
end


function calculate_confidence_interval_simple(b, n, SE, α=0.05)
    model = TDist(n - 2)
    t_ = quantile.(model, [α / 2, 1 - (α / 2)])  # returns a vector with both quantiles
    correction = t_ * SE

    # return a vector containing the confidence interval
    b .+ correction
end

