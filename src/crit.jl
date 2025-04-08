

function mean_field(x::Array{T,4}) where T<:Union{Float64, ComplexF64}
    x1 = 0.0
    x2 = 0.0
    x3 = 0.0
    x4 = 0.0
    x5 = 0.0
    x6 = 0.0

    _, N1, N2, N3 = size(x)
    @inbounds begin
        for i3 in 1:N3
            for i2 in 1:N2
                for i1 in 1:N1
                    x1 += x[1, i1, i2, i3]
                    x2 += x[2, i1, i2, i3]
                    x3 += x[3, i1, i2, i3]
                    x4 += x[4, i1, i2, i3]
                    x5 += x[5, i1, i2, i3]
                    x6 += x[6, i1, i2, i3]
                end
            end
        end
    end
    return [x1, x2, x3, x4, x5, x6] / (N1 * N2 * N3)
end





function eq_err(sig::Array{T,4}) where T<:Union{Float64, ComplexF64}
    err = 0.0
    _, N1, N2, N3 = size(sig)
    @inbounds begin
        for k in 1:N3
            for j in 1:N2
                for i in 1:N1
                    err += sig[1, i, j, k]^2 + sig[2, i, j, k]^2 + sig[3, i, j, k]^2 + 2 * sig[4, i, j, k]^2 + 2 * sig[5, i, j, k]^2 + 2 * sig[6, i, j, k]^2
                end
            end
        end
    end
    return abs(err) / (N1 * N2 * N3)
end