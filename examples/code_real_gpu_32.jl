using CUDA
using CUDA.CUFFT
using Adapt

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
Adapt.@adapt_structure IE

function IE(kappa, mu)
    lambda = kappa - 2 / 3 * mu
    return IE(kappa, mu, lambda)
end

# ************************************************************
# * RDC

function compute_sig!(sig1, sig2, sig3, sig4, sig5, sig6, eps1, eps2, eps3, eps4, eps5, eps6, phases, material_list)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if i <= length(eps1)
        mat = material_list[phases[i]]
        tre = eps1[i] + eps2[i] + eps3[i]

        sig1[i] = tre * mat.lambda + eps1[i] * 2 * mat.mu
        sig2[i] = tre * mat.lambda + eps2[i] * 2 * mat.mu
        sig3[i] = tre * mat.lambda + eps3[i] * 2 * mat.mu
        sig4[i] = eps4[i] * 2 * mat.mu
        sig5[i] = eps5[i] * 2 * mat.mu
        sig6[i] = eps6[i] * 2 * mat.mu
    end
    return nothing
end

# ************************************************************
# * gamma0

function gamma0_kernel!(tau1, tau2, tau3, tau4, tau5, tau6, c0, xi1, xi2, xi3, N, coef5, mu0, cartesian)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if i <= length(tau1)

        i1 = cartesian[i][1]
        i2 = cartesian[i][2]
        i3 = cartesian[i][3]

        divt1 = xi1[i1] * tau1[i] + xi2[i2] * tau6[i] + xi3[i3] * tau5[i]
        divt2 = xi1[i1] * tau6[i] + xi2[i2] * tau2[i] + xi3[i3] * tau4[i]
        divt3 = xi1[i1] * tau5[i] + xi2[i2] * tau4[i] + xi3[i3] * tau3[i]

        
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
            tau1[i] = 0.0
            tau2[i] = 0.0
            tau3[i] = 0.0
            tau4[i] = 0.0
            tau5[i] = 0.0
            tau6[i] = 0.0
        else
            xisquare = xi1[i1] * xi1[i1] + xi2[i2] * xi2[i2] + xi3[i3] * xi3[i3]
            dd = xi1[i1] * divt1 + xi2[i2] * divt2 + xi3[i3] * divt3

            fu1 = 1.0im * (divt1 / xisquare / mu0 - xi1[i1] * coef5 * dd / (xisquare * xisquare))
            fu2 = 1.0im * (divt2 / xisquare / mu0 - xi2[i2] * coef5 * dd / (xisquare * xisquare))
            fu3 = 1.0im * (divt3 / xisquare / mu0 - xi3[i3] * coef5 * dd / (xisquare * xisquare))


            tau1[i] = 1.0im * xi1[i1] * fu1
            tau2[i] = 1.0im * xi2[i2] * fu2
            tau3[i] = 1.0im * xi3[i3] * fu3
            tau4[i] = 1.0im * 0.5 * (xi2[i2] * fu3 + xi3[i3] * fu2)
            tau5[i] = 1.0im * 0.5 * (xi1[i1] * fu3 + xi3[i3] * fu1)
            tau6[i] = 1.0im * 0.5 * (xi1[i1] * fu2 + xi2[i2] * fu1)
        end


    end
    return nothing
end


function gamma0!(P, Pinv, xi1, xi2, xi3, tau1, tau2, tau3, tau4, tau5, tau6, sig1, sig2, sig3, sig4, sig5, sig6, c0)
    CUDA.CUFFT.mul!(tau1, P, sig1)
    CUDA.CUFFT.mul!(tau2, P, sig2)
    CUDA.CUFFT.mul!(tau3, P, sig3)
    CUDA.CUFFT.mul!(tau4, P, sig4)
    CUDA.CUFFT.mul!(tau5, P, sig5)
    CUDA.CUFFT.mul!(tau6, P, sig6)

    N=size(sig1)
    N = N |> cu
    coef5 = (c0.lambda + c0.mu) / (c0.mu * (c0.lambda + 2.0 * c0.mu))
    mu0 = c0.mu

    cartesian = CartesianIndices(size(tau1))

    n_blocks, n_threads = get_blocks_threads(tau1)
    @cuda blocks = n_blocks threads = n_threads gamma0_kernel!(tau1, tau2, tau3, tau4, tau5, tau6, c0, xi1, xi2, xi3, N, coef5, mu0, cartesian)

    CUDA.@allowscalar tau1[1, 1, 1] = 0.0
    CUDA.@allowscalar tau2[1, 1, 1] = 0.0
    CUDA.@allowscalar tau3[1, 1, 1] = 0.0
    CUDA.@allowscalar tau4[1, 1, 1] = 0.0
    CUDA.@allowscalar tau5[1, 1, 1] = 0.0
    CUDA.@allowscalar tau6[1, 1, 1] = 0.0

    CUDA.CUFFT.mul!(sig1, Pinv, tau1)
    CUDA.CUFFT.mul!(sig2, Pinv, tau2)
    CUDA.CUFFT.mul!(sig3, Pinv, tau3)
    CUDA.CUFFT.mul!(sig4, Pinv, tau4)
    CUDA.CUFFT.mul!(sig5, Pinv, tau5)
    CUDA.CUFFT.mul!(sig6, Pinv, tau6)

end




# **********************************************************************************************
# * FFT variables
function init_gpu_fft(N, M, L)

    A = CUDA.rand(Float32, N, M, L)
    P = CUDA.CUFFT.plan_rfft(A)
    Pinv = CUDA.CUFFT.plan_irfft(P * A, N)

    xi1 = CUDA.CUFFT.rfftfreq(N, N)
    xi2 = CUDA.CUFFT.fftfreq(M, M)
    xi3 = CUDA.CUFFT.fftfreq(L, L)
    
    tau1 = CUDA.zeros(ComplexF32, div(N, 2) + 1, M, L)
    tau2 = CUDA.zeros(ComplexF32, div(N, 2) + 1, M, L)
    tau3 = CUDA.zeros(ComplexF32, div(N, 2) + 1, M, L)
    tau4 = CUDA.zeros(ComplexF32, div(N, 2) + 1, M, L)
    tau5 = CUDA.zeros(ComplexF32, div(N, 2) + 1, M, L)
    tau6 = CUDA.zeros(ComplexF32, div(N, 2) + 1, M, L)

    return P, Pinv, xi1, xi2, xi3, tau1, tau2, tau3, tau4, tau5, tau6
end

function init_fields(N1, N2, N3)
    return CUDA.zeros(Float32, N1, N2, N3),
    CUDA.zeros(Float32, N1, N2, N3),
    CUDA.zeros(Float32, N1, N2, N3),
    CUDA.zeros(Float32, N1, N2, N3),
    CUDA.zeros(Float32, N1, N2, N3),
    CUDA.zeros(Float32, N1, N2, N3),
    CUDA.zeros(Float32, N1, N2, N3),
    CUDA.zeros(Float32, N1, N2, N3),
    CUDA.zeros(Float32, N1, N2, N3),
    CUDA.zeros(Float32, N1, N2, N3),
    CUDA.zeros(Float32, N1, N2, N3),
    CUDA.zeros(Float32, N1, N2, N3)
end

function get_blocks_threads(x)
    threads_per_block = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    n_blocks = cld(length(x), threads_per_block)
    return n_blocks, threads_per_block
end


function update_strain!(eps1, eps2, eps3, eps4, eps5, eps6, sig1, sig2, sig3, sig4, sig5, sig6)
    CUDA.@. eps1 = eps1 + sig1
    CUDA.@. eps2 = eps2 + sig2
    CUDA.@. eps3 = eps3 + sig3
    CUDA.@. eps4 = eps4 + sig4
    CUDA.@. eps5 = eps5 + sig5
    CUDA.@. eps6 = eps6 + sig6
end

# **********************************************************************************************
# * Error
function eq_error!(r, sig1, sig2, sig3, sig4, sig5, sig6)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    @inbounds if i <= length(r)
        r[i] = sig1[i]^2 + sig2[i]^2 + sig3[i]^2 + 2 * sig4[i]^2 + 2 * sig5[i]^2 + 2 * sig6[i]^2
        # r[i] = sig1[i]*sig1[i] + sig2[i]*sig2[i] + sig3[i]*sig3[i] + 2 * sig4[i]*sig4[i] + 2 * sig5[i]*sig5[i] + 2 * sig6[i]*sig6[i]
    end
    return nothing
end

function eq_error(r,sig1, sig2, sig3, sig4, sig5, sig6)
    n_blocks, n_threads = get_blocks_threads(sig1)
    @cuda blocks = n_blocks threads = n_threads eq_error!(r, sig1, sig2, sig3, sig4, sig5, sig6)
    residu = reduce(+, r)
    residu = residu / length(sig1)
end


function means(eps1, eps2, eps3, eps4, eps5, eps6, sig1, sig2, sig3, sig4, sig5, sig6)
    Nlength = length(eps1)
    eps1_m = reduce(+, eps1) / Nlength
    eps2_m = reduce(+, eps2) / Nlength
    eps3_m = reduce(+, eps3) / Nlength
    eps4_m = reduce(+, eps4) / Nlength
    eps5_m = reduce(+, eps5) / Nlength
    eps6_m = reduce(+, eps6) / Nlength
    sig1_m = reduce(+, sig1) / Nlength
    sig2_m = reduce(+, sig2) / Nlength
    sig3_m = reduce(+, sig3) / Nlength
    sig4_m = reduce(+, sig4) / Nlength
    sig5_m = reduce(+, sig5) / Nlength
    sig6_m = reduce(+, sig6) / Nlength
    return [eps1_m, eps2_m, eps3_m, eps4_m, eps5_m, eps6_m], [sig1_m, sig2_m, sig3_m, sig4_m, sig5_m, sig6_m]
end


# **********************************************************************************************
# * Main
function main()
    material_list = [IE(10.0, 5.0), IE(2.0, 1.0)]
    c0 = chose_c0(material_list)

    N = 128
    N1, N2, N3 = N, N, N
    # N1, N2, N3 = 1024, 1024, 1
    phases = generate_micro(N1, N2, N3)

    # *********
    phases_gpu = cu(phases)
    material_list_gpu = [m |> cu for m in material_list] |> cu

    P, Pinv, xi1, xi2, xi3, tau1, tau2, tau3, tau4, tau5, tau6 = init_gpu_fft(N1, N2, N3)

    #*------
    eps1, eps2, eps3, eps4, eps5, eps6, sig1, sig2, sig3, sig4, sig5, sig6 = init_fields(N1, N2, N3)
    CUDA.@. eps1 += 1.0
    n_blocks, n_threads = get_blocks_threads(eps1)
    @cuda blocks = n_blocks threads = n_threads compute_sig!(sig1, sig2, sig3, sig4, sig5, sig6, eps1, eps2, eps3, eps4, eps5, eps6, phases_gpu, material_list_gpu)
    
    r = CUDA.zeros(Float32, size(eps1))
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
        while  it < it_max # &&err > tol 
            it += 1

            tgamma0 = CUDA.@elapsed gamma0!(P, Pinv, xi1, xi2, xi3, tau1, tau2, tau3, tau4, tau5, tau6, sig1, sig2, sig3, sig4, sig5, sig6, c0)

            tmajeps = CUDA.@elapsed update_strain!(eps1, eps2, eps3, eps4, eps5, eps6, sig1, sig2, sig3, sig4, sig5, sig6)

            terr = CUDA.@elapsed err = eq_error(r,sig1, sig2, sig3, sig4, sig5, sig6)

            tsig = CUDA.@elapsed @cuda blocks = n_blocks threads = n_threads compute_sig!(sig1, sig2, sig3, sig4, sig5, sig6, eps1, eps2, eps3, eps4, eps5, eps6, phases_gpu, material_list_gpu)

            tmean = CUDA.@elapsed EPS, SIG = means(eps1, eps2, eps3, eps4, eps5, eps6, sig1, sig2, sig3, sig4, sig5, sig6)

            
            println("$it $err $EPS $SIG ")
            chrono_gamma0 += tgamma0
            chrono_majeps += tmajeps
            chrono_sig += tsig
            chrono_err += terr
            chrono_mean += tmean
        end
    end

    println("Temps total: $tit")
    println("chrono_gamma0 (+ fft + ifft) = $chrono_gamma0")
    println("chrono_majeps = $chrono_majeps")
    println("chrono_sig0 = $chrono_sig")
    println("chrono_err = $chrono_err")
    println("chrono_mean = $chrono_mean")

    return
end

main()
# @profview main()