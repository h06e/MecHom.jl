using FFTW

struct FFTInfo
    xi::Array{Frequencies{Float64},1}
    P::AbstractFFTs.Plan
    Pinv::AbstractFFTs.Plan
    n::Vector{Int64}
end


function FFTInfo(x::Array{T}) where {T<:Union{Float64,ComplexF64}}
    dims = ndims(x)
    sizes = size(x)

    if T == Float64
        # Generate frequency arrays
        xi = Vector{Frequencies{Float64}}(undef, dims - 1)
        xi[1] = rfftfreq(sizes[2], sizes[2])
        xi[2:end] .= [fftfreq(sizes[i], sizes[i]) for i in 3:dims]

        # FFT setup
        dim_fft = Tuple(i for i = 2:dims)
        P = plan_rfft(x, dim_fft; flags=FFTW.MEASURE)
        tau = P * x
        Pinv = plan_irfft(tau, sizes[2], dim_fft; flags=FFTW.MEASURE)
    else
        # Generate frequency arrays
        xi = Vector{Frequencies{Float64}}(undef, dims - 1)
        xi[1:end] .= [fftfreq(sizes[i], sizes[i]) for i in 2:dims]

        # FFT setup
        dim_fft = Tuple(i for i = 2:dims)
        P = plan_fft(x, dim_fft; flags=FFTW.MEASURE)
        tau = P * x
        Pinv = plan_ifft(tau, dim_fft; flags=FFTW.MEASURE)
    end
    # Return FFTInfo structure
    return FFTInfo(xi, P, Pinv, collect(sizes[2:end])), tau
end



#*********************************************************************************
#* GPU
#*********************************************************************************


function init_gpu_realfft(::Type{T},eps, p, phases) where {T<:AbstractFloat}
    N = size(phases)
    C = Complex{T}
    tau = CUDA.zeros(C, div(N[1], 2) + 1, N[2], N[3], 6)
    
    #? FFT plans
    P = CUDA.CUFFT.plan_rfft(eps, (1, 2, 3))
    Pinv = CUDA.CUFFT.plan_irfft(tau, N[1], (1, 2, 3))

    xi1 = CUDA.zeros(div(N[1], 2) + 1)
    xi2 = CUDA.zeros(N[2])
    xi3 = CUDA.zeros(N[3])
    xi1 .= CUDA.CUFFT.rfftfreq(N[1], 1 / p[1])
    xi2 .= CUDA.CUFFT.fftfreq(N[2], 1 / p[2])
    xi3 .= CUDA.CUFFT.fftfreq(N[3], 1 / p[3])

    return P, Pinv, xi1, xi2, xi3, tau
end


function init_gpu_complexfft(::Type{T},eps, p, phases) where {T}
    N = size(phases)
    tau = CUDA.zeros(T, N[1], N[2], N[3], 6)

    #? FFT plans
    P = CUDA.CUFFT.plan_fft(eps, (1, 2, 3))
    Pinv = CUDA.CUFFT.plan_ifft(tau, (1, 2, 3))

    xi1 = CUDA.zeros(N[1])
    xi2 = CUDA.zeros(N[2])
    xi3 = CUDA.zeros(N[3])
    xi1 .= CUDA.CUFFT.fftfreq(N[1], 1 / p[1])
    xi2 .= CUDA.CUFFT.fftfreq(N[2], 1 / p[2])
    xi3 .= CUDA.CUFFT.fftfreq(N[3], 1 / p[3])

    return P, Pinv, xi1, xi2, xi3, tau
end


function modified_frequencied(::Type{T}, p, phases) where {T}
    if T <: Real
        N = size(phases)
        dx, dy, dz = p

        T1 = N[1] * dx
        T2 = N[2] * dy
        T3 = N[3] * dz

        DF1 = 2 * pi / T1
        DF2 = 2 * pi / T2
        DF3 = 2 * pi / T3

        filter_radius = 1.0

        ros2 = filter_radius / 2

        FREQ = zeros(T, div(N[1], 2) + 1, N[2], N[3], 3)

        x1 = rfftfreq(N[1], 1 / p[1])
        x2 = fftfreq(N[2], 1 / p[2])
        x3 = fftfreq(N[3], 1 / p[3])

        for k in 1:N[3]
            for j in 1:N[2]
                for i in 1:div(N[1], 2)+1


                    U1 = x1[i] * DF1 
                    U2 = x2[j] * DF2 
                    U3 = x3[k] * DF3 

                    FREQ[i, j, k, 1] = (N[1] / (ros2 * T1)) * sin(ros2 * U1 * T1 / N[1]) * cos(ros2 * U2 * T2 / N[2]) * cos(ros2 * U3 * T3 / N[3])
                    FREQ[i, j, k, 2] = (N[2] / (ros2 * T2)) * cos(ros2 * U1 * T1 / N[1]) * sin(ros2 * U2 * T2 / N[2]) * cos(ros2 * U3 * T3 / N[3])
                    FREQ[i, j, k, 3] = (N[3] / (ros2 * T3)) * cos(ros2 * U1 * T1 / N[1]) * cos(ros2 * U2 * T2 / N[2]) * sin(ros2 * U3 * T3 / N[3])

                    FREQ[i, j, k, 1] = 2/dx * sin(x1[i]*2*pi/2) * cos(x2[j]*2*pi/2) * cos(x3[k]*2*pi/2)
                    FREQ[i, j, k, 2] = 2/dy * cos(x1[i]*2*pi/2) * sin(x2[j]*2*pi/2) * cos(x3[k]*2*pi/2)
                    FREQ[i, j, k, 3] = 2/dz * cos(x1[i]*2*pi/2) * cos(x2[j]*2*pi/2) * sin(x3[k]*2*pi/2)
                    # FREQ[i,j,k,1] = U1
                    # FREQ[i,j,k,2] = U2
                    # FREQ[i,j,k,3] = U3

                end
            end
        end

        freq = CUDA.zeros(T, div(N[1], 2) + 1, N[2], N[3], 3)
        copyto!(freq, FREQ)
        return freq
    else
        N = size(phases)
        dx, dy, dz = p

        T1 = N[1] * dx
        T2 = N[2] * dy
        T3 = N[3] * dz

        DF1 = 2 * pi / T1
        DF2 = 2 * pi / T2
        DF3 = 2 * pi / T3

        filter_radius = 1.0

        ros2 = filter_radius / 2

        FREQ = zeros(T, N[1], N[2], N[3], 3)

        x1 = fftfreq(N[1], 1 / p[1])
        x2 = fftfreq(N[2], 1 / p[2])
        x3 = fftfreq(N[3], 1 / p[3])

        for k in 1:N[3]
            for j in 1:N[2]
                for i in 1:N[1]

                    U1 = x1[i] * DF1 
                    U2 = x2[j] * DF2 
                    U3 = x3[k] * DF3 

                    FREQ[i, j, k, 1] = (N[1] / (ros2 * T1)) * sin(ros2 * U1 * T1 / N[1]) * cos(ros2 * U2 * T2 / N[2]) * cos(ros2 * U3 * T3 / N[3])
                    FREQ[i, j, k, 2] = (N[2] / (ros2 * T2)) * cos(ros2 * U1 * T1 / N[1]) * sin(ros2 * U2 * T2 / N[2]) * cos(ros2 * U3 * T3 / N[3])
                    FREQ[i, j, k, 3] = (N[3] / (ros2 * T3)) * cos(ros2 * U1 * T1 / N[1]) * cos(ros2 * U2 * T2 / N[2]) * sin(ros2 * U3 * T3 / N[3])

                end
            end
        end

        freq = CUDA.zeros(T, N[1], N[2], N[3], 3)
        copyto!(freq, FREQ)

        return freq
    end
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


function add_mean_value_kernel!(eps, mean_value, NNN, cartesian)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    @inbounds if i <= NNN
        i1 = cartesian[i][1]
        i2 = cartesian[i][2]
        i3 = cartesian[i][3]

        eps[i1, i2, i3, 1] += mean_value[1]
        eps[i1, i2, i3, 2] += mean_value[2]
        eps[i1, i2, i3, 3] += mean_value[3]
        eps[i1, i2, i3, 4] += mean_value[4]
        eps[i1, i2, i3, 5] += mean_value[5]
        eps[i1, i2, i3, 6] += mean_value[6]
    end
    return nothing
end


function add_mean_value!(eps, mean_value, cartesian)

    mean_value_gpu = cu(mean_value)
    NNN = size(eps, 1) * size(eps, 2) * size(eps, 3)
    n_blocks, n_threads = get_blocks_threads(NNN)
    @cuda blocks = n_blocks threads = n_threads add_mean_value_kernel!(eps, mean_value_gpu, NNN, cartesian)


end