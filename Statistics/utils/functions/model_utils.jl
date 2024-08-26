function hand_predict_simple(model, data=nothing)
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

function hand_predict(model, data=nothing)
    if data === nothing
        data = model.X
    end

    _, predictors = termnames(model.formula)

    coefs = Vector{Float64}()
    if predictors isa String
        intercept = get(model.coefs, "Intercept", nothing)
        coef = get(model.coefs, predictors, nothing)
        push!(coefs, coef)

        return coefs .* data .+ intercept
    else
        predictors[1] = "Intercept"
        for predictor in predictors
            coef = get(model.coefs, predictor, nothing)
            push!(coefs, coef)
        end

        return data * coefs
    end


end
