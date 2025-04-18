using CUDA
using CUDA.CUFFT
using Adapt
using FFTW

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

function compute_sig!(sig, eps, phases, material_list,cartesian,nelmt)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if i <= nelmt

        i1 = cartesian[i][1]
        i2 = cartesian[i][2]
        i3 = cartesian[i][3]


        mat = material_list[phases[i]]
        tre = eps[i1,i2,i3,1] + eps[i1,i2,i3,2] + eps[i1,i2,i3,3]

        sig[i1,i2,i3,1] = eps[i1,i2,i3,1] * 2 * mat.mu + tre * mat.lambda
        sig[i1,i2,i3,2] = eps[i1,i2,i3,2] * 2 * mat.mu + tre * mat.lambda
        sig[i1,i2,i3,3] = eps[i1,i2,i3,3] * 2 * mat.mu + tre * mat.lambda
        sig[i1,i2,i3,4] = eps[i1,i2,i3,4] * 2 * mat.mu
        sig[i1,i2,i3,5] = eps[i1,i2,i3,5] * 2 * mat.mu
        sig[i1,i2,i3,6] = eps[i1,i2,i3,6] * 2 * mat.mu
    end
    return nothing
end

# ************************************************************
# * gamma0

function gamma0_kernel!(tau, ntau, freq, N, coef5, mu0, cartesian)

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


function gamma0!(P, Pinv, freq, tau, sig, c0)
    CUDA.CUFFT.mul!(tau, P, sig)

    N = [size(sig, 1), size(sig, 2), size(sig, 3)]
    N = N |> cu
    coef5 = (c0.lambda + c0.mu) / (c0.mu * (c0.lambda + 2.0 * c0.mu))
    mu0 = c0.mu

    cartesian = CartesianIndices(size(tau[:, :, :, 1]))

    # mean_value = [0.,0.,0.,0.,0.,0.]

    ntau = size(tau, 1) * size(tau, 2) * size(tau, 3)
    n_blocks, n_threads = get_blocks_threads(ntau)
    @cuda blocks = n_blocks threads = n_threads gamma0_kernel!(tau, ntau, freq, N, coef5, mu0, cartesian)

    CUDA.@allowscalar tau[1, 1, 1, 1] = 0.0
    CUDA.@allowscalar tau[1, 1, 1, 2] = 0.0
    CUDA.@allowscalar tau[1, 1, 1, 3] = 0.0
    CUDA.@allowscalar tau[1, 1, 1, 4] = 0.0
    CUDA.@allowscalar tau[1, 1, 1, 5] = 0.0
    CUDA.@allowscalar tau[1, 1, 1, 6] = 0.0

    CUDA.CUFFT.mul!(sig, Pinv, tau)

end




# **********************************************************************************************
# * FFT variables

function initFreq_hexa!(FREQ, grid, fft_start, fft_end, filter_radius)
    
end


function init_gpu_fft(phases, dxyz)
    N = size(phases)

    #? FFT plans
    A = CUDA.rand(Float32, N[1], N[2], N[3],6)
    P = CUDA.CUFFT.plan_rfft(A, (1,2,3))
    Pinv = CUDA.CUFFT.plan_irfft(P * A, N[1], (1,2,3))

    #? Modified frequencies for Willot Green Operator


    dx, dy, dz = dxyz

    T1 = N[1] * dx
    T2 = N[2] * dy
    T3 = N[3] * dz

    DF1 = 2*pi / T1
    DF2 = 2*pi / T2
    DF3 = 2*pi / T3

    filter_radius = 1.0

    ros2 = filter_radius / 2

    FREQ = zeros(Float32, div(N[1], 2) + 1, N[2], N[3],3)

    x1 = rfftfreq(N[1],1/dxyz[1])
    x2 = fftfreq(N[2],1/dxyz[2])
    x3 = fftfreq(N[3],1/dxyz[3])

    for k in 1:N[3]
        for j in 1:N[2]
            for i in 1:div(N[1], 2) + 1


                U1 = x1[i] * DF1
                U2 = x2[j] * DF2
                U3 = x3[k] * DF3

                FREQ[i,j,k,1] = (N[1] / (ros2 * T1)) * sin(ros2 * U1 * T1 / N[1]) * cos(ros2 * U2 * T2 / N[2]) * cos(ros2 * U3 * T3 / N[3])
                FREQ[i,j,k,2] = (N[2] / (ros2 * T2)) * cos(ros2 * U1 * T1 / N[1]) * sin(ros2 * U2 * T2 / N[2]) * cos(ros2 * U3 * T3 / N[3])
                FREQ[i,j,k,3] = (N[3] / (ros2 * T3)) * cos(ros2 * U1 * T1 / N[1]) * cos(ros2 * U2 * T2 / N[2]) * sin(ros2 * U3 * T3 / N[3])
                # FREQ[i,j,k,1] = U1
                # FREQ[i,j,k,2] = U2
                # FREQ[i,j,k,3] = U3
            end
        end
    end

    println(FREQ[:,1,1,1])

    freq = CUDA.zeros(Float32, div(N[1], 2) + 1, N[2], N[3],3)
    copyto!(freq, FREQ) 
    
    
    tau = CUDA.zeros(ComplexF32, div(N[1], 2) + 1, N[2], N[3],6)

    return P, Pinv, freq, tau
end

function get_blocks_threads(x)
    threads_per_block = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    n_blocks = cld(length(x), threads_per_block)
    return n_blocks, threads_per_block
end


function get_blocks_threads(N::Int64)
    threads_per_block = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    n_blocks = cld(N, threads_per_block)
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


function eq_error!(r, sig, cartesian,nelmt)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    @inbounds if i <= nelmt

        i1 = cartesian[i][1]
        i2 = cartesian[i][2]
        i3 = cartesian[i][3]


        r[i] = sig[i1,i2,i3,1]*sig[i1,i2,i3,1] + sig[i1,i2,i3,2]*sig[i1,i2,i3,2] + sig[i1,i2,i3,3]*sig[i1,i2,i3,3] + 2 * sig[i1,i2,i3,4]*sig[i1,i2,i3,4] + 2 * sig[i1,i2,i3,5]*sig[i1,i2,i3,5] + 2 * sig[i1,i2,i3,6]*sig[i1,i2,i3,6]
        # r[i] = sig1[i]*sig1[i] + sig2[i]*sig2[i] + sig3[i]*sig3[i] + 2 * sig4[i]*sig4[i] + 2 * sig5[i]*sig5[i] + 2 * sig6[i]*sig6[i]
    end
    return nothing
end

function eq_error(r,sig,cartesian)
    nelmt = size(sig,1)*size(sig,2)*size(sig,3)
    n_blocks, n_threads = get_blocks_threads(nelmt)
    @cuda blocks = n_blocks threads = n_threads eq_error!(r, sig, cartesian,nelmt)
    residu = reduce(+, r)
    residu = residu / nelmt
end



function eq_error(a)
    result = CUDA.sum(
        a[:, :, :, 1] .^ 2 +
        a[:, :, :, 2] .^ 2 +
        a[:, :, :, 3] .^ 2 +
        2.0f0 .* a[:, :, :, 4] .^ 2 +
        2.0f0 .* a[:, :, :, 5] .^ 2 +
        2.0f0 .* a[:, :, :, 6] .^ 2
    ) / (size(a, 1) * size(a, 2) * size(a, 3))
    return result
end

function meanfield(x)

    Nlength = size(x, 1) * size(x, 2) * size(x, 3)

    sums = CUDA.sum(reshape(x, :, size(x, 4)), dims=1)
    X = vec(sums) ./ Nlength
    return Array(X)
end


# **********************************************************************************************
# * Main
function main()
    material_list = [IE(1.0, 1.0), IE(1e3, 1e3)]
    c0 = chose_c0(material_list)

    N = 32
    N1, N2, N3 = N, N, N
    N1, N2, N3 = 512, 512, 1
    NNN = N1*N2*N3
    phases = generate_micro(N1, N2, N3)

    # *********
    phases_gpu = cu(phases)
    material_list_gpu = [m |> cu for m in material_list] |> cu

    # init_gpu_fft(phases, [1/N1, 1/N2, 1/N3])
    # return
    P, Pinv, freq, tau = init_gpu_fft(phases, [1/N1, 1/N2, 1/N3])
    #*------
    eps = CUDA.zeros(Float32, N1, N2, N3, 6) 
    sig = CUDA.zeros(Float32, N1, N2, N3, 6)

    cartesian = CartesianIndices(size(phases))

    CUDA.fill!(view(eps, :, :, :, 1), 1.0f0)

    n_blocks, n_threads = get_blocks_threads(NNN)
    @cuda blocks = n_blocks threads = n_threads compute_sig!(sig, eps, phases_gpu, material_list_gpu,cartesian,NNN)

    EPS = meanfield(eps)
    SIG = meanfield(sig)


    r = CUDA.zeros(Float32, N1, N2, N3) 

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

            tgamma0 = CUDA.@elapsed gamma0!(P, Pinv, freq, tau, sig, c0)

            tmajeps = CUDA.@elapsed CUDA.@. eps .+= sig

            # terr = CUDA.@elapsed err = eq_error(sig)
            terr = CUDA.@elapsed err = eq_error(r, sig, cartesian)

            tsig = CUDA.@elapsed @cuda blocks = n_blocks threads = n_threads compute_sig!(sig, eps, phases_gpu, material_list_gpu, cartesian,NNN)

            tmean = CUDA.@elapsed begin
                EPS .= meanfield(eps)
                SIG .= meanfield(sig)
                
            end 

            
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