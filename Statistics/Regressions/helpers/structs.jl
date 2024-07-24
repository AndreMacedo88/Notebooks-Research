using Printf
using DataFrames
using StatsModels

"""
    LinearModelOLS(formula::FormulaTerm, data::DataFrames.DataFrame)

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
struct LinearModelOLS
    formula::FormulaTerm
    data::DataFrames.DataFrame
    coefs::Dict
    
    function LinearModelOLS(formula::FormulaTerm, data::DataFrames.DataFrame)
        y, x = _process_formula(formula, data)
        b0, b1 = _estimate_params_OLS(y, x)

        _, predictor = termnames(formula)

        coefs = Dict{String, Real}("Intercept" => b0, predictor => b1)  # works only for one predictor

        new(formula, data, coefs)

    end

end

function _process_formula(formula, data)
    schema_data = apply_schema(formula, schema(formula, data))
    modelcols(schema_data, data)
end

"""
Estimation of the parameters by OLS
"""
function _estimate_params_OLS(y, x)
    n = length(x)
        
    x̄ = (1/n) .* sum(x)
    ȳ = (1/n) .* sum(y)

    cov = ((1/n) .* sum(y.*x)) - ((1/(n^2)) .* sum(y) .* sum(x))
    var_x = ((1/n) .* sum(x.^2)) - (((1/n) .* sum(x))^2)

    b1 = cov / var_x
    b0 = ȳ - (b1 * x̄)

    b0, b1

end

function _significance_test_parameters()
end

function hand_predict(model, data=nothing)
    if data === nothing
        data = model.data
    end

    _, predictors = termnames(model.formula)

    intercept = get(model.coefs, "Intercept", nothing)

    coefs = Vector{Float64}()
    if predictors isa String
        coef = get(model.coefs, predictors, nothing)
        push!(coefs, coef)
    else
        for predictor in predictors
            coef = get(model.coefs, predictor, nothing)
            push!(coefs, coef)
        end
    end

    data_predictors = data[:, predictors]
    coefs .* data_predictors .+ intercept

end

function Base.show(io::IO, model::LinearModelOLS)
    @printf("%s\n\n", "OLS Linear Regression Model")
    @printf("%-10s%s\n\n\n", "Formula:", model.formula)
    
    b0 = get(model.coefs, "Intercept", nothing)
    _, predictors = termnames(model.formula)

    @printf("%15s:\n", "Coefficients")
    @printf("%s\n", "────────────────────────────────────")
    @printf("%26s\n", "Coef.")
    @printf("%s\n", "────────────────────────────────────")
    @printf("%15s:%10.3f\n", "Intercept", b0)

    if predictors isa String
        value = get(model.coefs, predictors, nothing)
        @printf("%15s:%10.3f\n", predictors, value)
        
    else
        for term in predictors
            value = round(get(model.coefs, term, nothing), digits=3)
            @printf("%15s:%10s\n", term, string(value))
        end
    end
    @printf("%s\n", "────────────────────────────────────")
end
