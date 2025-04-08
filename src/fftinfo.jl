using FFTW

struct FFTInfo
    xi::Array{Frequencies{Float64},1}
    P::AbstractFFTs.Plan
    Pinv::AbstractFFTs.Plan
    n::Vector{Int64}
end


function FFTInfo(x::Array{T}) where T<:Union{Float64, ComplexF64}
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


function init_gpu_realfft(::Type{T}, N, M, L) where {T<:AbstractFloat}
    A = CUDA.rand(T, N, M, L)
    P = CUDA.CUFFT.plan_rfft(A)
    Pinv = CUDA.CUFFT.plan_irfft(P * A, N)

    xi1 = CUDA.CUFFT.rfftfreq(N, N)
    xi2 = CUDA.CUFFT.fftfreq(M, M)
    xi3 = CUDA.CUFFT.fftfreq(L, L)

    C = Complex{T}
    shape = (div(N, 2) + 1, M, L)
    tau1 = CUDA.zeros(C, shape...)
    tau2 = CUDA.zeros(C, shape...)
    tau3 = CUDA.zeros(C, shape...)
    tau4 = CUDA.zeros(C, shape...)
    tau5 = CUDA.zeros(C, shape...)
    tau6 = CUDA.zeros(C, shape...)
    return P, Pinv, xi1, xi2, xi3, tau1, tau2, tau3, tau4, tau5, tau6
end

function init_gpu_complexfft(::Type{T}, N, M, L) where T
    A = CUDA.rand(T, N, M, L)
    P = CUDA.CUFFT.plan_fft(A)
    Pinv = CUDA.CUFFT.plan_ifft(P * A)

    xi1 = CUDA.CUFFT.fftfreq(N, N)
    xi2 = CUDA.CUFFT.fftfreq(M, M)
    xi3 = CUDA.CUFFT.fftfreq(L, L)

    shape = (N, M, L)
    tau1 = CUDA.zeros(T, shape...)
    tau2 = CUDA.zeros(T, shape...)
    tau3 = CUDA.zeros(T, shape...)
    tau4 = CUDA.zeros(T, shape...)
    tau5 = CUDA.zeros(T, shape...)
    tau6 = CUDA.zeros(T, shape...)
    return P, Pinv, xi1, xi2, xi3, tau1, tau2, tau3, tau4, tau5, tau6
end


function init_fields(::Type{T},N1, N2, N3) where T
    return CUDA.zeros(T, N1, N2, N3),
    CUDA.zeros(T, N1, N2, N3),
    CUDA.zeros(T, N1, N2, N3),
    CUDA.zeros(T, N1, N2, N3),
    CUDA.zeros(T, N1, N2, N3),
    CUDA.zeros(T, N1, N2, N3),
    CUDA.zeros(T, N1, N2, N3),
    CUDA.zeros(T, N1, N2, N3),
    CUDA.zeros(T, N1, N2, N3),
    CUDA.zeros(T, N1, N2, N3),
    CUDA.zeros(T, N1, N2, N3),
    CUDA.zeros(T, N1, N2, N3)
end



function get_blocks_threads(x)
    threads_per_block = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    n_blocks = cld(length(x), threads_per_block)
    return n_blocks, threads_per_block
end
