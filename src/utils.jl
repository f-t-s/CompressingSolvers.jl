import LinearAlgebra: Factorization, norm 

function compute_relative_error(L::SparseMatrixCSC, A::Factorization, max_iter=100, number_trials=1)
    x = randn(size(L, 1), number_trials)
    x = x / norm(x)
    for k = 1 : max_iter
        x .= L * (L' * x) - A \ x
        x .= x ./ mapslices(norm, x, dims=1)
        # @show x' * (L * (L' * x) - A \ x)
    end
    err = sum(mapslices(norm, x' * (L * (L' * x) - A \ x), dims=1)) / number_trials
    x = randn(size(L, 1), number_trials)
    x = x / norm(x)
    for k = 1 : max_iter
        x .= A \ x
        x .= x ./ mapslices(norm, x, dims=1)
        # @show x' * (A \ x)
    end
    @show nrm = sum(mapslices(norm, x' * (A \ x), dims = 1)) / number_trials
    return abs(err) / nrm
end
