using FFTW
using CUDA
using CUDA.CUFFT

"Isotropic green operator (CPU)"
function gamma0!(tau::Array{ComplexF64}, sig::Array{T}, c0::IE, fftinfo::FFTInfo; mean_value::Vector) where {T<:Union{Float64,ComplexF64}}

    FFTW.mul!(tau, fftinfo.P, sig)

    xi1 = fftinfo.xi[1]
    xi2 = fftinfo.xi[2]
    xi3 = fftinfo.xi[3]
    _, N1, N2, N3 = size(tau)

    divt = zeros(ComplexF64, 3)
    fu = zeros(ComplexF64, 3)

    coef5 = (c0.lambda + c0.mu) / (c0.mu * (c0.lambda + 2.0 * c0.mu))
    @inbounds begin
        for i3 in 1:N3
            for i2 in 1:N2
                for i1 in 1:N1

                    divt[1] = xi1[i1] * tau[1, i1, i2, i3] + xi2[i2] * tau[6, i1, i2, i3] + xi3[i3] * tau[5, i1, i2, i3]
                    divt[2] = xi1[i1] * tau[6, i1, i2, i3] + xi2[i2] * tau[2, i1, i2, i3] + xi3[i3] * tau[4, i1, i2, i3]
                    divt[3] = xi1[i1] * tau[5, i1, i2, i3] + xi2[i2] * tau[4, i1, i2, i3] + xi3[i3] * tau[3, i1, i2, i3]

                    ic = 0
                    if fftinfo.n[1] % 2 == 0 && i1 == div(fftinfo.n[1], 2) + 1
                        ic = ic + 1
                    end
                    if fftinfo.n[2] % 2 == 0 && i2 == div(fftinfo.n[2], 2) + 1
                        ic = ic + 1
                    end
                    if fftinfo.n[3] % 2 == 0 && i3 == div(fftinfo.n[3], 2) + 1
                        ic = ic + 1
                    end

                    if ic != 0
                        tau[1, i1, i2, i3] = 0.0
                        tau[2, i1, i2, i3] = 0.0
                        tau[3, i1, i2, i3] = 0.0
                        tau[4, i1, i2, i3] = 0.0
                        tau[5, i1, i2, i3] = 0.0
                        tau[6, i1, i2, i3] = 0.0
                    else
                        xisquare = xi1[i1] * xi1[i1] + xi2[i2] * xi2[i2] + xi3[i3] * xi3[i3]
                        dd = xi1[i1] * divt[1] + xi2[i2] * divt[2] + xi3[i3] * divt[3]

                        fu[1] = 1.0im * (divt[1] / xisquare / c0.mu - xi1[i1] * coef5 * dd / (xisquare * xisquare))
                        fu[2] = 1.0im * (divt[2] / xisquare / c0.mu - xi2[i2] * coef5 * dd / (xisquare * xisquare))
                        fu[3] = 1.0im * (divt[3] / xisquare / c0.mu - xi3[i3] * coef5 * dd / (xisquare * xisquare))


                        tau[1, i1, i2, i3] = 1.0im * xi1[i1] * fu[1]
                        tau[2, i1, i2, i3] = 1.0im * xi2[i2] * fu[2]
                        tau[3, i1, i2, i3] = 1.0im * xi3[i3] * fu[3]
                        tau[4, i1, i2, i3] = 1.0im * 0.5 * (xi2[i2] * fu[3] + xi3[i3] * fu[2])
                        tau[5, i1, i2, i3] = 1.0im * 0.5 * (xi1[i1] * fu[3] + xi3[i3] * fu[1])
                        tau[6, i1, i2, i3] = 1.0im * 0.5 * (xi1[i1] * fu[2] + xi2[i2] * fu[1])
                    end
                end
            end
        end
    end

    tau[:, 1, 1, 1] .= mean_value .* (length(sig) / 6)

    FFTW.mul!(sig, fftinfo.Pinv, tau)

end




"Isotropic green operator (CPU)"
function gamma0!(tau::Array{ComplexF64}, sig::Array{T}, c0::ITE, fftinfo::FFTInfo; mean_value::Vector) where {T<:Union{Float64,ComplexF64}}

    FFTW.mul!(tau, fftinfo.P, sig)

    xi1 = fftinfo.xi[1]
    xi2 = fftinfo.xi[2]
    xi3 = fftinfo.xi[3]
    _, N1, N2, N3 = size(tau)

    divt = zeros(ComplexF64, 3)
    fu = zeros(ComplexF64, 3)

    α = c0.k + c0.m
    αp = c0.m
    β = c0.n
    γ = c0.p
    γp = c0.p + c0.l

    @inbounds begin
        for i3 in 1:N3
            for i2 in 1:N2
                for i1 in 1:N1

                    divt[1] = xi1[i1] * tau[1, i1, i2, i3] + xi2[i2] * tau[6, i1, i2, i3] + xi3[i3] * tau[5, i1, i2, i3]
                    divt[2] = xi1[i1] * tau[6, i1, i2, i3] + xi2[i2] * tau[2, i1, i2, i3] + xi3[i3] * tau[4, i1, i2, i3]
                    divt[3] = xi1[i1] * tau[5, i1, i2, i3] + xi2[i2] * tau[4, i1, i2, i3] + xi3[i3] * tau[3, i1, i2, i3]

                    ic = 0
                    if fftinfo.n[1] % 2 == 0 && i1 == div(fftinfo.n[1], 2) + 1
                        ic = ic + 1
                    end
                    if fftinfo.n[2] % 2 == 0 && i2 == div(fftinfo.n[2], 2) + 1
                        ic = ic + 1
                    end
                    if fftinfo.n[3] % 2 == 0 && i3 == div(fftinfo.n[3], 2) + 1
                        ic = ic + 1
                    end

                    if ic != 0
                        tau[1, i1, i2, i3] = 0.0
                        tau[2, i1, i2, i3] = 0.0
                        tau[3, i1, i2, i3] = 0.0
                        tau[4, i1, i2, i3] = 0.0
                        tau[5, i1, i2, i3] = 0.0
                        tau[6, i1, i2, i3] = 0.0
                    else
                        η2 = xi1[i1] * xi1[i1] + xi2[i2] * xi2[i2]
                        ξ12 = xi1[i1] * xi1[i1]
                        ξ22 = xi2[i2] * xi2[i2]
                        ξ32 = xi3[i3] * xi3[i3]

                        D = (αp * η2 + γ * ξ32) *
                               (α * γ * η2 * η2 + (α * β + γ * γ - γp * γp) * η2 * ξ32 + β * γ * ξ32 * ξ32)

                        N11 = (αp * ξ12 + α * ξ22 + γ * ξ32) *
                                 (γ * η2 + β * ξ32) - γp * γp * ξ22 * ξ32
                        N12 = γp * γp * xi1[i1] * xi2[i2] * ξ32 - (α - αp) * xi1[i1] * xi2[i2] * (γ * η2 + β * ξ32)
                        N13 = (α - αp) * γp * xi1[i1] * ξ22 * xi3[i3] - γp * xi1[i1] * xi3[i3] * (αp * ξ12 + α * ξ22 + γ * ξ32)

                        N22 = (α * ξ12 + αp * ξ22 + γ * ξ32) * (γ * η2 + β * ξ32) - γp * γp * ξ12 * ξ32
                        N23 = (α - αp) * γp * ξ12 * xi2[i2] * xi3[i3] - γp * xi2[i2] * xi3[i3] * (α * ξ12 + αp * ξ22 + γ * ξ32)

                        N33 = (α * ξ12 + αp * ξ22 + γ * ξ32) *
                                 (αp * ξ12 + α * ξ22 + γ * ξ32) - (α - αp) * (α - αp) * ξ12 * ξ22

                        fu[1] = 1.0im * (N11 * divt[1] + N12 * divt[2] + N13 * divt[3]) / D
                        fu[2] = 1.0im * (N12 * divt[1] + N22 * divt[2] + N23 * divt[3]) / D
                        fu[3] = 1.0im * (N13 * divt[1] + N23 * divt[2] + N33 * divt[3]) / D


                        tau[1, i1, i2, i3] = 1.0im * xi1[i1] * fu[1]
                        tau[2, i1, i2, i3] = 1.0im * xi2[i2] * fu[2]
                        tau[3, i1, i2, i3] = 1.0im * xi3[i3] * fu[3]
                        tau[4, i1, i2, i3] = 1.0im * 0.5 * (xi2[i2] * fu[3] + xi3[i3] * fu[2])
                        tau[5, i1, i2, i3] = 1.0im * 0.5 * (xi1[i1] * fu[3] + xi3[i3] * fu[1])
                        tau[6, i1, i2, i3] = 1.0im * 0.5 * (xi1[i1] * fu[2] + xi2[i2] * fu[1])
                    end

                end
            end
        end
    end
    tau[:, 1, 1, 1] .= mean_value .* (length(sig) / 6)

    FFTW.mul!(sig, fftinfo.Pinv, tau)
end


#***********************************************************************************************



function gamma0IE_kernel!(tau, ntau, xi1, xi2, xi3, N, coef5, mu0, cartesian)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if i <= ntau

        i1 = cartesian[i][1]
        i2 = cartesian[i][2]
        i3 = cartesian[i][3]

        divt1 = xi1[i1] * tau[i1, i2, i3, 1] + xi2[i2] * tau[i1, i2, i3, 6] + xi3[i3] * tau[i1, i2, i3, 5]
        divt2 = xi1[i1] * tau[i1, i2, i3, 6] + xi2[i2] * tau[i1, i2, i3, 2] + xi3[i3] * tau[i1, i2, i3, 4]
        divt3 = xi1[i1] * tau[i1, i2, i3, 5] + xi2[i2] * tau[i1, i2, i3, 4] + xi3[i3] * tau[i1, i2, i3, 3]

        ic = 0
        if N[1] % 2 == 0 && i1 == div(N[1], 2) + 1
            ic = ic + 1
        end
        if N[2] % 2 == 0 && i2 == div(N[2], 2) + 1
            ic = ic + 1
        end
        if N[3] % 2 == 0 && i3 == div(N[3], 2) + 1
            ic = ic + 1
        end

        if ic != 0
            tau[i1, i2, i3, 1] = 0.0
            tau[i1, i2, i3, 2] = 0.0
            tau[i1, i2, i3, 3] = 0.0
            tau[i1, i2, i3, 4] = 0.0
            tau[i1, i2, i3, 5] = 0.0
            tau[i1, i2, i3, 6] = 0.0
        else
            xisquare = xi1[i1] * xi1[i1] + xi2[i2] * xi2[i2] + xi3[i3] * xi3[i3]
            dd = xi1[i1] * divt1 + xi2[i2] * divt2 + xi3[i3] * divt3

            fu1 = 1.0im * (divt1 / xisquare / mu0 - xi1[i1] * coef5 * dd / (xisquare * xisquare))
            fu2 = 1.0im * (divt2 / xisquare / mu0 - xi2[i2] * coef5 * dd / (xisquare * xisquare))
            fu3 = 1.0im * (divt3 / xisquare / mu0 - xi3[i3] * coef5 * dd / (xisquare * xisquare))


            tau[i1, i2, i3, 1] = 1.0im * xi1[i1] * fu1
            tau[i1, i2, i3, 2] = 1.0im * xi2[i2] * fu2
            tau[i1, i2, i3, 3] = 1.0im * xi3[i3] * fu3
            tau[i1, i2, i3, 4] = 1.0im * 0.5 * (xi2[i2] * fu3 + xi3[i3] * fu2)
            tau[i1, i2, i3, 5] = 1.0im * 0.5 * (xi1[i1] * fu3 + xi3[i3] * fu1)
            tau[i1, i2, i3, 6] = 1.0im * 0.5 * (xi1[i1] * fu2 + xi2[i2] * fu1)
        end


    end
    return nothing
end

function gamma0IE_willot_kernel!(tau, ntau, freq, N, coef5, mu0, cartesian)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if i <= ntau #!!! @inbounds 

        i1 = cartesian[i][1]
        i2 = cartesian[i][2]
        i3 = cartesian[i][3]

        xi1 = freq[i1,i2,i3,1]
        xi2 = freq[i1,i2,i3,2]
        xi3 = freq[i1,i2,i3,3]

        divt1 = xi1 * tau[i1, i2, i3, 1] + xi2 * tau[i1, i2, i3, 6] + xi3 * tau[i1, i2, i3, 5]
        divt2 = xi1 * tau[i1, i2, i3, 6] + xi2 * tau[i1, i2, i3, 2] + xi3 * tau[i1, i2, i3, 4]
        divt3 = xi1 * tau[i1, i2, i3, 5] + xi2 * tau[i1, i2, i3, 4] + xi3 * tau[i1, i2, i3, 3]


        ic = 0
        if N[1] % 2 == 0 && i1 == div(N[1], 2) + 1
            ic = ic + 1
        end
        if N[2] % 2 == 0 && i2 == div(N[2], 2) + 1
            ic = ic + 1
        end
        if N[3] % 2 == 0 && i3 == div(N[3], 2) + 1
            ic = ic + 1
        end

        if ic != 0
            tau[i1, i2, i3, 1] = 0.0
            tau[i1, i2, i3, 2] = 0.0
            tau[i1, i2, i3, 3] = 0.0
            tau[i1, i2, i3, 4] = 0.0
            tau[i1, i2, i3, 5] = 0.0
            tau[i1, i2, i3, 6] = 0.0
        else
            xisquare = xi1 * xi1 + xi2 * xi2 + xi3 * xi3
            dd = xi1 * divt1 + xi2 * divt2 + xi3 * divt3

            fu1 = 1.0im * (divt1 / xisquare / mu0 - xi1 * coef5 * dd / (xisquare * xisquare))
            fu2 = 1.0im * (divt2 / xisquare / mu0 - xi2 * coef5 * dd / (xisquare * xisquare))
            fu3 = 1.0im * (divt3 / xisquare / mu0 - xi3 * coef5 * dd / (xisquare * xisquare))


            tau[i1, i2, i3, 1] = 1.0im * xi1 * fu1
            tau[i1, i2, i3, 2] = 1.0im * xi2 * fu2
            tau[i1, i2, i3, 3] = 1.0im * xi3 * fu3
            tau[i1, i2, i3, 4] = 1.0im * 0.5 * (xi2 * fu3 + xi3 * fu2)
            tau[i1, i2, i3, 5] = 1.0im * 0.5 * (xi1 * fu3 + xi3 * fu1)
            tau[i1, i2, i3, 6] = 1.0im * 0.5 * (xi1 * fu2 + xi2 * fu1)
        end


    end
    return nothing
end



function gamma0ITE_kernel!(tau,ntau, xi1, xi2, xi3, N, α, αp, β, γ, γp, cartesian)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if i <= ntau

        i1 = cartesian[i][1]
        i2 = cartesian[i][2]
        i3 = cartesian[i][3]

        divt1 = xi1[i1] * tau[i1, i2, i3, 1] + xi2[i2] * tau[i1, i2, i3, 6] + xi3[i3] * tau[i1, i2, i3, 5]
        divt2 = xi1[i1] * tau[i1, i2, i3, 6] + xi2[i2] * tau[i1, i2, i3, 2] + xi3[i3] * tau[i1, i2, i3, 4]
        divt3 = xi1[i1] * tau[i1, i2, i3, 5] + xi2[i2] * tau[i1, i2, i3, 4] + xi3[i3] * tau[i1, i2, i3, 3]

        ic = 0
        if N[1] % 2 == 0 && i1 == div(N[1], 2) + 1
            ic = ic + 1
        end
        if N[2] % 2 == 0 && i2 == div(N[2], 2) + 1
            ic = ic + 1
        end
        if N[3] % 2 == 0 && i3 == div(N[3], 2) + 1
            ic = ic + 1
        end

        if ic != 0
            tau[i1, i2, i3, 1] = 0.0
            tau[i1, i2, i3, 2] = 0.0
            tau[i1, i2, i3, 3] = 0.0
            tau[i1, i2, i3, 4] = 0.0
            tau[i1, i2, i3, 5] = 0.0
            tau[i1, i2, i3, 6] = 0.0
        else
        
            η2 = xi1[i1] * xi1[i1] + xi2[i2] * xi2[i2]
            ξ12 = xi1[i1] * xi1[i1]
            ξ22 = xi2[i2] * xi2[i2]
            ξ32 = xi3[i3] * xi3[i3]

            D = (αp * η2 + γ * ξ32) *
                    (α * γ * η2 * η2 + (α * β + γ * γ - γp * γp) * η2 * ξ32 + β * γ * ξ32 * ξ32)

            N11 = (αp * ξ12 + α * ξ22 + γ * ξ32) *
                        (γ * η2 + β * ξ32) - γp * γp * ξ22 * ξ32
            N12 = γp * γp * xi1[i1] * xi2[i2] * ξ32 - (α - αp) * xi1[i1] * xi2[i2] * (γ * η2 + β * ξ32)
            N13 = (α - αp) * γp * xi1[i1] * ξ22 * xi3[i3] - γp * xi1[i1] * xi3[i3] * (αp * ξ12 + α * ξ22 + γ * ξ32)

            N22 = (α * ξ12 + αp * ξ22 + γ * ξ32) * (γ * η2 + β * ξ32) - γp * γp * ξ12 * ξ32
            N23 = (α - αp) * γp * ξ12 * xi2[i2] * xi3[i3] - γp * xi2[i2] * xi3[i3] * (α * ξ12 + αp * ξ22 + γ * ξ32)

            N33 = (α * ξ12 + αp * ξ22 + γ * ξ32) *
                        (αp * ξ12 + α * ξ22 + γ * ξ32) - (α - αp) * (α - αp) * ξ12 * ξ22

            fu1 = 1.0im * (N11 * divt1 + N12 * divt2 + N13 * divt3) / D
            fu2 = 1.0im * (N12 * divt1 + N22 * divt2 + N23 * divt3) / D
            fu3 = 1.0im * (N13 * divt1 + N23 * divt2 + N33 * divt3) / D


            tau[i1, i2, i3, 1] = 1.0im * xi1 * fu1
            tau[i1, i2, i3, 2] = 1.0im * xi2 * fu2
            tau[i1, i2, i3, 3] = 1.0im * xi3 * fu3
            tau[i1, i2, i3, 4] = 1.0im * 0.5 * (xi2 * fu3 + xi3 * fu2)
            tau[i1, i2, i3, 5] = 1.0im * 0.5 * (xi1 * fu3 + xi3 * fu1)
            tau[i1, i2, i3, 6] = 1.0im * 0.5 * (xi1 * fu2 + xi2 * fu1)

        end

    end
    return nothing
end

function gamma0!(P, Pinv, xi1, xi2, xi3, tau, sig, c0, mean, freq_mod)

    CUDA.CUFFT.mul!(tau, P, sig)

    N = size(sig)
    N = N |> cu
    NNN =  N[1]*N[2]*N[3]

    ntau = size(tau, 1) * size(tau, 2) * size(tau, 3)
    cartesian = CartesianIndices(size(tau[:, :, :, 1]))
    n_blocks, n_threads = get_blocks_threads(ntau)


    if isnothing(freq_mod)
        if c0 isa IE
            coef5 = (c0.lambda + c0.mu) / (c0.mu * (c0.lambda + 2.0 * c0.mu))
            mu0 = c0.mu
            @cuda blocks = n_blocks threads = n_threads gamma0IE_kernel!(tau, ntau,xi1, xi2, xi3, N, coef5, mu0, cartesian)
        elseif c0 isa ITE
            α = c0.k + c0.m
            αp = c0.m
            β = c0.n
            γ = c0.p
            γp = c0.p + c0.l
            @cuda blocks = n_blocks threads = n_threads gamma0ITE_kernel!(tau,ntau, xi1, xi2, xi3, N, α, αp, β, γ, γp, cartesian)
        end
    else
        coef5 = Float32((c0.lambda + c0.mu) / (c0.mu * (c0.lambda + 2.0 * c0.mu)))
        mu0 = Float32(c0.mu)

        @cuda blocks = n_blocks threads = n_threads gamma0IE_willot_kernel!(tau, ntau, freq_mod, N, coef5, mu0, cartesian)
    end

    CUDA.@allowscalar tau[1, 1, 1, 1] = mean[1] * NNN
    CUDA.@allowscalar tau[1, 1, 1, 2] = mean[2] * NNN
    CUDA.@allowscalar tau[1, 1, 1, 3] = mean[3] * NNN
    CUDA.@allowscalar tau[1, 1, 1, 4] = mean[4] * NNN
    CUDA.@allowscalar tau[1, 1, 1, 5] = mean[5] * NNN
    CUDA.@allowscalar tau[1, 1, 1, 6] = mean[6] * NNN
    

    CUDA.CUFFT.mul!(sig, Pinv, tau)
end






