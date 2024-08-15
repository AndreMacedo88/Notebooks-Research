using Printf
using DataFrames
using StatsModels
using Logging

"""
    LinearModelSimple(formula::FormulaTerm, data::DataFrames.DataFrame)

Compute a simple Linear Regression

# Method
- The parameters are estimated via Ordinary Least Squares (OLS)

OLS is an analytical solution that is equivalent to the MLE for a certain type 
of linear regression:
- The model is correctly specified:
    - We have not omitted important variables in the model (underfitting the data)
    - We do not have redundant variables in the model (overfitting the data)
    - The necessary transformations of the variables are applied (e.g. to linearize the relation to the response variable in the case of linear regressions)
    - We do not have outliers in the residuals of the model
- The residuals are independent and identically distributed
- The explanatory variables are not correlated with anything but the response variable

The OLS equations are derived from minimizing the Sum of Squared Errors (SSE) (see proof [here](https://openforecast.org/sba/OLS.html))

For the linear regression, the OLS estimator produces the Best Linear Unbiased Estimates (BLUE), 
as shown by the Gauss Markov theorem. 
Explanation [here](https://www.statlect.com/fundamentals-of-statistics/Gauss-Markov-theorem).

"""
struct LinearModelSimple
    formula::FormulaTerm
    data::DataFrames.DataFrame
    coefs::Dict
    residuals::Matrix
    SST::Real
    SSE::Real
    SSR::Real
    MSE::Real
    SE::Dict
    t::Dict
    pval::Dict
    ci::Dict
    R²::Real

    function LinearModelSimple(formula::FormulaTerm, data::DataFrames.DataFrame)
        y, x = _process_formula(formula, data)
        n = length(x)
        SST = _SST(y)

        b0, b1 = _estimate_params_OLS_simple(y, x, n)

        e = _errors_simple(b0, b1, y, x)
        SSE = _SSE(e)
        MSE = _MSE(SSE, n)
        residual_se = √(MSE)

        SSR = SST - SSE
        R² = SSR / SST

        SE0, SE1 = [_SE(residual_se, x, n, type) for type in ["intercept", "predictor"]]
        t0, t1 = [_t_statistic_parameters(coef, SE) for (coef, SE) in zip([b0, b1], [SE0, SE1])]
        pval0, pval1 = [_significance_test_parameters(t, n) for t in [t0, t1]]
        ci0, ci1 = [_confidence_interval(coef, n, SE) for (coef, SE) in zip([b0, b1], [SE0, SE1])]

        _, predictor = termnames(formula)

        # the following assignments work only for one predictor
        coefs = Dict{String,Real}("Intercept" => b0, predictor => b1)
        SEs = Dict{String,Real}("Intercept" => SE0, predictor => SE1)
        ts = Dict{String,Real}("Intercept" => t0, predictor => t1)
        pvals = Dict{String,Real}("Intercept" => pval0, predictor => pval1)
        cis = Dict{String,Vector}("Intercept" => ci0, predictor => ci1)

        new(
            formula,
            data,
            coefs,
            e,
            SST,
            SSE,
            SSR,
            MSE,
            SEs,
            ts,
            pvals,
            cis,
            R²
        )

    end

end


"""
    LinearModelOLS(formula::FormulaTerm, data::DataFrames.DataFrame)

Compute a Multiple Linear Regression

# Method
- The parameters are estimated via Ordinary Least Squares (OLS)

OLS is an analytical solution that is equivalent to the MLE for a certain type 
of linear regression:
- The model is correctly specified:
    - We have not omitted important variables in the model (underfitting the data)
    - We do not have redundant variables in the model (overfitting the data)
    - The necessary transformations of the variables are applied (e.g. to linearize the relation to the response variable in the case of linear regressions)
    - We do not have outliers in the residuals of the model
- The residuals are independent and identically distributed
- The explanatory variables are not correlated with anything but the response variable

The OLS equations are derived from minimizing the Sum of Squared Errors (SSE) (see proof [here](https://openforecast.org/sba/OLS.html))

For the linear regression, the OLS estimator produces the Best Linear Unbiased Estimates (BLUE), 
as shown by the Gauss Markov theorem. 
Explanation [here](https://www.statlect.com/fundamentals-of-statistics/Gauss-Markov-theorem).

"""
struct LinearModelOLS
    formula::FormulaTerm
    data::DataFrames.DataFrame
    coefs::Dict
    residuals::Matrix
    SST::Real
    SSE::Real
    SSR::Real
    MSE::Real
    SE::Dict
    t::Dict
    pval::Dict
    ci::Dict
    R²::Real

    function LinearModelOLS(formula::FormulaTerm, data::DataFrames.DataFrame)
        y, X = _process_formula(formula, data)
        @info("X:", size(X))
        n = size(X)[1]
        df = n - size(X)[2]
        SST = _SST(y)

        b = _estimate_params_OLS(y, X)
        @info("Parameter estimation:", b)

        e = _errors(b, y, X)
        SSE = _SSE(e)
        MSE = _MSE(SSE, df)
        RMSE = √(MSE)  # or standard error

        SSR = SST - SSE
        R² = _R_adjusted(SSE, SST, n, df)

        @info("Metrics:", e, SSE, MSE, RMSE, SSR, R²)

        SE0, SE1 = [_SE(RMSE, X, n, type) for type in ["intercept", "predictor"]]
        t0, t1 = [_t_statistic_parameters(coef, SE) for (coef, SE) in zip(b, [SE0, SE1])]
        pval0, pval1 = [_significance_test_parameters(t, n) for t in [t0, t1]]
        ci0, ci1 = [_confidence_interval(coef, n, SE) for (coef, SE) in zip(b, [SE0, SE1])]
        @info("More metrics:", SE0, SE1, t0, t1, pval0, pval1, ci0, ci1)

        _, predictor = termnames(formula)
        @info("predictor: ", predictor)

        # the following assignments work only for one predictor
        coefs = Dict{String,Real}("Intercept" => b0, predictor => b1)
        SEs = Dict{String,Real}("Intercept" => SE0, predictor => SE1)
        ts = Dict{String,Real}("Intercept" => t0, predictor => t1)
        pvals = Dict{String,Real}("Intercept" => pval0, predictor => pval1)
        cis = Dict{String,Vector}("Intercept" => ci0, predictor => ci1)

        new(
            formula,
            data,
            coefs,
            e,
            SST,
            SSE,
            SSR,
            MSE,
            SEs,
            ts,
            pvals,
            cis,
            R²
        )

    end

end

function _process_formula(formula, data)
    schema_data = apply_schema(formula, schema(formula, data))
    modelcols(schema_data, data)
end


"""
Estimation of the parameters by OLS
"""
function _estimate_params_OLS_simple(y, x, n)
    x̄ = (1 / n) .* sum(x)
    ȳ = (1 / n) .* sum(y)

    cov = ((1 / n) .* sum(y .* x)) - ((1 / (n^2)) .* sum(y) .* sum(x))
    var_x = ((1 / n) .* sum(x .^ 2)) - (((1 / n) .* sum(x))^2)

    b1 = cov / var_x
    b0 = ȳ - (b1 * x̄)

    b0, b1
end

"""
Estimation of the parameters by OLS
"""
function _estimate_params_OLS(y, X)
    β = inv(transpose(X) * X) * (transpose(X) * y)
    β
end

_errors_simple(intercept, coefs, y, x) = y .- (coefs .* x .+ intercept)

# _errors(coefs, y, x) = y .- (coefs[2:end] .* x .+ coefs[1])

_errors(coefs, y, X) = y .- (coefs' * X')

_SSE(e) = sum(e_ -> e_^2, e) # anonymous to avoid allocating space to the squared values

function _MSE(SSE, df)
    SSE / df
end

function _SE(residual_se, x, n, type)
    X̄ = sum(x) / n

    if type == "intercept"
        return residual_se * √((1 / n) + ((X̄ .^ 2) / sum((x .- X̄) .^ 2)))
    elseif type == "predictor"
        return residual_se / √(sum((x .- X̄) .^ 2))
    else
        @warn "The type of variable does not exist. Unable to calculate SE." type
    end
end

_t_statistic_parameters(coef, SE) = coef / SE

function _significance_test_parameters(t, n)
    df = n - 2
    model = TDist(df)

    # return 2-sided p-value
    (1 - cdf(model, abs(t)) + cdf(model, -abs(t)))
end

function _confidence_interval(b, n, SE, α=0.05)
    model = TDist(n - 2)
    t_ = quantile.(model, [α / 2, 1 - (α / 2)])  # returns a vector with both quantiles
    correction = t_ * SE

    # return a vector containing the confidence interval
    b .+ correction
end

function _SST(y)
    ȳ = mean(y)
    sum((y .- ȳ) .^ 2)
end

function _R_adjusted(SSE, SST, n, df)
    1 - (((n - 1) * SSE) / (df * SST))
end

function Base.show(io::IO, model::LinearModelOLS)
    @printf("%s\n\n", "OLS Linear Regression Model")
    @printf("%-10s%s\n\n\n", "Formula:", model.formula)

    b0 = get(model.coefs, "Intercept", nothing)
    SE0 = get(model.SE, "Intercept", nothing)
    t0 = get(model.t, "Intercept", nothing)
    pval0 = get(model.pval, "Intercept", nothing)
    ci0 = get(model.ci, "Intercept", nothing)
    ci0 = round.(ci0; digits=2)

    _, predictors = termnames(model.formula)

    @printf("%15s:\n", "Coefficients")
    @printf("%s\n", "─────────────────────────────────────────────────────────────────────────────")
    @printf(
        "%26s%12s%10s%10s%18s\n",
        "Coef.",
        "Std.Error",
        "t",
        "pval",
        "95% CI"
    )
    @printf("%s\n", "─────────────────────────────────────────────────────────────────────────────")
    @printf(
        "%15s:%10.3f%12.2f%10.2f%10.0e%18s\n",
        "Intercept",
        b0,
        SE0,
        t0,
        pval0,
        ci0
    )

    if predictors isa String
        coef = get(model.coefs, predictors, nothing)
        SE = get(model.SE, predictors, nothing)
        t = get(model.t, predictors, nothing)
        pval = get(model.pval, predictors, nothing)
        ci = get(model.ci, predictors, nothing)
        ci = round.(ci; digits=2)
        @printf(
            "%15s:%10.3f%12.2f%10.2f%10.0e%18s\n",
            predictors,
            coef,
            SE,
            t,
            pval,
            ci
        )

    else
        for term in predictors
            coef = get(model.coefs, term, nothing)
            SE = get(model.SE, term, nothing)
            t = get(model.t, term, nothing)
            pval = get(model.pval, term, nothing)
            ci = get(model.ci, term, nothing)
            ci = round.(ci; digits=2)
            @printf(
                "%15s:%10.3f%12.2f%10.2f%10.3e%18s\n",
                term,
                coef,
                SE,
                t,
                pval,
                ci
            )
        end
    end
    @printf("%s\n", "─────────────────────────────────────────────────────────────────────────────")
end

