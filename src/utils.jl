using LinearAlgebra: Factorization

function compute_relative_error(L::SparseMatrixCSC, A::Factorization, max_iter=100)
    x = randn(size(L, 1))
    x = x / norm(x)
    for k = 1 : max_iter
        x .= L * (L' * x) - A \ x
        x .= x / norm(x)
        # @show x' * (L * (L' * x) - A \ x)
    end
    err = x' * (L * (L' * x) - A \ x)
    x = randn(size(L, 1))
    x = x / norm(x)
    for k = 1 : max_iter
        x .= A \ x
        x .= x / norm(x)
        # @show x' * (A \ x)
    end
    @show nrm = x' * (A \ x)
    return abs(err) / nrm
end
