using FFTW
# using Profile

"Génère une microstructure avec une inclusion sphérique"
function generate_micro(N1, N2, N3)
    phases = ones(Int32, N1, N2, N3)
    r2 = (N1 / 4)^2
    @inbounds begin
        for k in 1:N3
            for j in 1:N2
                for i in 1:N1
                    if ((i - N1 / 2)^2 + (j - N2 / 2)^2 + (k - N3 / 3)^2) < r2
                        phases[i, j, k] = 2
                    end
                end
            end
        end
    end
    return phases
end


function chose_c0(material_list)
    klist = [mat.kappa for mat in material_list]
    mlist = [mat.mu for mat in material_list]
    kmin = minimum(klist)
    kmax = maximum(klist)
    mmin = minimum(mlist)
    mmax = maximum(mlist)

    return IE(0.5 * (kmin + kmax), 0.5 * (mmin + mmax))
end

# ************************************************************
# * Behaviors

struct IE
    kappa::Float64
    mu::Float64
    lambda::Float64
end

function IE(kappa, mu)
    lambda = kappa - 2 / 3 * mu
    return IE(kappa, mu, lambda)
end

# ************************************************************
# * RDC

function rdc!(sig, eps, phases, material_list)
    _, N1, N2, N3 = size(eps)
    @inbounds begin
        for k in 1:N3
            for j in 1:N2
                for i in 1:N1
                    mat = material_list[phases[i, j, k]]
                    lambda = mat.lambda
                    mu = mat.mu

                    tre = eps[1, i, j, k] + eps[2, i, j, k] + eps[3, i, j, k]

                    sig[1, i, j, k] = tre * lambda + eps[1, i, j, k] * 2 * mu
                    sig[2, i, j, k] = tre * lambda + eps[2, i, j, k] * 2 * mu
                    sig[3, i, j, k] = tre * lambda + eps[3, i, j, k] * 2 * mu
                    sig[4, i, j, k] = eps[4, i, j, k] * 2 * mu
                    sig[5, i, j, k] = eps[5, i, j, k] * 2 * mu
                    sig[6, i, j, k] = eps[6, i, j, k] * 2 * mu
                end
            end
        end
    end
end

# ************************************************************
# * gamma0

function gamma0!(tau, sig, c0, fftinfo)

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

    # tau[1, 1, 1, 1] = 0.0
    # tau[2, 1, 1, 1] = 0.0
    # tau[3, 1, 1, 1] = 0.0
    # tau[4, 1, 1, 1] = 0.0
    # tau[5, 1, 1, 1] = 0.0
    # tau[6, 1, 1, 1] = 0.0

    tau[:, 1, 1, 1] .= 0.0

    FFTW.mul!(sig, fftinfo.Pinv, tau)
end

# ************************************************************
# * fftinfo

struct FFTInfo
    xi::Array{Frequencies{Float64},1}
    P::AbstractFFTs.Plan
    Pinv::AbstractFFTs.Plan
    n::Vector{Int64}
end


function FFTInfo(x::Array{Float64})
    dims = ndims(x)
    sizes = size(x)

    # Generate frequency arrays
    xi = Vector{Frequencies{Float64}}(undef, dims - 1)
    xi[1] = rfftfreq(sizes[2], sizes[2])
    xi[2:end] .= [fftfreq(sizes[i], sizes[i]) for i in 3:dims]

    # FFT setup
    dim_fft = Tuple(i for i = 2:dims)
    P = plan_rfft(x, dim_fft; flags=FFTW.MEASURE)
    tau = P * x
    Pinv = plan_irfft(tau, sizes[2], dim_fft; flags=FFTW.MEASURE)

    # Return FFTInfo structure
    return FFTInfo(xi, P, Pinv, collect(sizes[2:end])), tau
end


function eq_err(sig)
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
    return err / (N1 * N2 * N3)
end



function mean(x)
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

function main()
    material_list = [IE(10.0, 5.0), IE(2.0, 1.0)]
    c0 = chose_c0(material_list)

    N = 128
    N1, N2, N3 = N, N, N
    N1, N2, N3 = 1024, 1024, 1
    phases = generate_micro(N1, N2, N3)

    E = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    eps = reshape(E, 6, 1, 1, 1) .+ zeros(Float64, 6, N1, N2, N3)

    sig = zeros(Float64, 6, N1, N2, N3)
    rdc!(sig, eps, phases, material_list)

    fftinfo, tau = FFTInfo(eps)


    tol = 1e-30
    it_max = 50

    err = 1.0
    it = 0

    chrono_gamma0 = 0.0
    chrono_majeps = 0.0
    chrono_sig = 0.0
    chrono_err = 0.0
    chrono_mean = 0.0

    tit = @elapsed begin
        while err > tol && it < it_max
            it += 1

            tgamma0 = @elapsed gamma0!(tau, sig, c0, fftinfo)

            tmajeps = @elapsed eps .+= sig

            terr = @elapsed err = eq_err(sig)

            tsig = @elapsed rdc!(sig, eps, phases, material_list)

            tmean = @elapsed EPS, SIG = mean(eps), mean(sig)

            # tmean = @elapsed et = sum(eps)
            
            println("$it $err $EPS $SIG ")
            chrono_gamma0 += tgamma0
            chrono_majeps += tmajeps
            chrono_sig += tsig
            chrono_err += terr
            chrono_mean += tmean
        end
    end

    println("Temps total: $tit")
    println("chrono_gamma0 (+ fft + fft-1) = $chrono_gamma0")
    println("chrono_majeps = $chrono_majeps")
    println("chrono_sig0 = $chrono_sig")
    println("chrono_err = $chrono_err")
    println("chrono_mean = $chrono_mean")

    return
end

main()
# @profview main()