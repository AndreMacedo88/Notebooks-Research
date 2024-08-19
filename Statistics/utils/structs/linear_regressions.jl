using Printf
using DataFrames
using StatsModels
using Logging
using Revise

includet("../functions/optimization_and_solvers.jl")
includet("../functions/model_statistics.jl")


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
        df = n - size(X)[2]  # in this model it will always be n-2

        SST = calculate_SST(y)

        b0, b1 = solve_normal_equation_simple(y, x, n)

        e = calculate_residuals_simple(b0, b1, y, x)
        SSE = calculate_SSE(e)
        MSE = calculate_MSE(SSE, n)
        residual_se = √(MSE)

        SSR = SST - SSE
        R² = SSR / SST

        SE0, SE1 = [calculate_SE_simple(residual_se, x, n, type) for type in ["intercept", "predictor"]]
        t0, t1 = [calculate_t_statistic(coef, SE) for (coef, SE) in zip([b0, b1], [SE0, SE1])]
        pval0, pval1 = [perform_ttest(t, df) for t in [t0, t1]]
        ci0, ci1 = [calculate_confidence_interval(coef, n, SE) for (coef, SE) in zip([b0, b1], [SE0, SE1])]

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
        SST = calculate_SST(y)
        @info("df:", df)

        coefs = solve_normal_equation(y, X)
        @info("Parameter estimation:", coefs)

        e = calculate_residuals(coefs, y, X)
        SSE = calculate_SSE(e)
        MSE = calculate_MSE(SSE, df)
        RMSE = √(MSE)  # or standard error

        SSR = SST - SSE
        R² = calculate_R_adjusted(SSE, SST, n, df)

        @info("Metrics:", e, SSE, MSE, RMSE, SSR, R²)
        SEs = calculate_SE_smallsamples(MSE, X, df)
        @info("", SEs)
        ts = calculate_t_statistic(coefs, SEs)
        @info("", ts)
        pvals = perform_ttest(ts, df)
        @info("", pvals)
        cis = calculate_confidence_interval(coef, df, SEs)
        @info("More metrics:", SEs, ts, pvals, cis)

        _, predictor = termnames(formula)
        @info("predictor: ", predictor)

        # the following assignments work only for one predictor
        # coefs = Dict{String,Real}("Intercept" => b0, predictor => b1)
        # SEs = Dict{String,Real}("Intercept" => SE0, predictor => SE1)
        # ts = Dict{String,Real}("Intercept" => t0, predictor => t1)
        # pvals = Dict{String,Real}("Intercept" => pval0, predictor => pval1)
        # cis = Dict{String,Vector}("Intercept" => ci0, predictor => ci1)

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

