using CUDA

export solverGPU

function update_strain!(eps1, eps2, eps3, eps4, eps5, eps6, sig1, sig2, sig3, sig4, sig5, sig6)
    CUDA.@. eps1 = eps1 + sig1
    CUDA.@. eps2 = eps2 + sig2
    CUDA.@. eps3 = eps3 + sig3
    CUDA.@. eps4 = eps4 + sig4
    CUDA.@. eps5 = eps5 + sig5
    CUDA.@. eps6 = eps6 + sig6
end

function eq_error!(r, sig1, sig2, sig3, sig4, sig5, sig6)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    @inbounds if i <= length(r)
        r[i] = sig1[i]^2 + sig2[i]^2 + sig3[i]^2 + 2 * sig4[i]^2 + 2 * sig5[i]^2 + 2 * sig6[i]^2
        # r[i] = sig1[i]*sig1[i] + sig2[i]*sig2[i] + sig3[i]*sig3[i] + 2 * sig4[i]*sig4[i] + 2 * sig5[i]*sig5[i] + 2 * sig6[i]*sig6[i]
    end
    return nothing
end

function eq_error(r, sig1, sig2, sig3, sig4, sig5, sig6)
    n_blocks, n_threads = get_blocks_threads(sig1)
    @cuda blocks = n_blocks threads = n_threads eq_error!(r, sig1, sig2, sig3, sig4, sig5, sig6)
    residu = reduce(+, r)
    residu = abs(residu) / length(sig1)
end


function means_gpu(eps1, eps2, eps3, eps4, eps5, eps6)
    Nlength = length(eps1)
    eps1_m = reduce(+, eps1) / Nlength
    eps2_m = reduce(+, eps2) / Nlength
    eps3_m = reduce(+, eps3) / Nlength
    eps4_m = reduce(+, eps4) / Nlength
    eps5_m = reduce(+, eps5) / Nlength
    eps6_m = reduce(+, eps6) / Nlength
    return [eps1_m, eps2_m, eps3_m, eps4_m, eps5_m, eps6_m]
end


function convert_output_fields(eps1, eps2, eps3, eps4, eps5, eps6, sig1, sig2, sig3, sig4, sig5, sig6)

    eps = zeros(Float64, 6, size(eps1)...)
    sig = zeros(Float64, 6, size(eps1)...)

    eps[1, :, :, :] .= Array(eps1)
    eps[2, :, :, :] .= Array(eps2)
    eps[3, :, :, :] .= Array(eps3)
    eps[4, :, :, :] .= Array(eps4)
    eps[5, :, :, :] .= Array(eps5)
    eps[6, :, :, :] .= Array(eps6)
    sig[1, :, :, :] .= Array(sig1)
    sig[2, :, :, :] .= Array(sig2)
    sig[3, :, :, :] .= Array(sig3)
    sig[4, :, :, :] .= Array(sig4)
    sig[5, :, :, :] .= Array(sig5)
    sig[6, :, :, :] .= Array(sig6)

    return eps, sig
end

function solverGPU(
    phases::Array{Int32},
    material_list::Vector{<:Material},
    loading_type::Type{<:LoadingType},
    loading_list::Vector{Vector{Float64}},
    time_list::Vector{T},
    tols::Vector{Float64};
    keep_it_info::Bool=false,
    verbose_fft::Bool=false,
    verbose_step::Bool=false,
    c0::Union{Nothing,<:Elastic,Symbol}=nothing,
    Nit_max::Int64=1000,
    scheme::Type{<:Scheme}=FixedPoint,
    polarization_AB::Vector{Float64}=[2.0, 2.0],
    polarization_skip_tests::Int64=0,
    keep_fields::Bool=false,
    save_fields::Bool=false,
    precision::Symbol=:double
) where {T<:Union{Float64,ComplexF64}}

    for mat in material_list
        if (mat isa ITE{ComplexF64} || mat isa IE{ComplexF64}) && T == Float64
            @error "Cannot have Complex material in material_list AND Real time_list --> convert time_list into Vector{ComplexF64}"
            throw(ArgumentError("Cannot have Complex material in material_list AND Real time_list --> convert time_list into Vector{ComplexF64}"))
        end
    end


    if isnothing(c0)
        c0 = chose_c0(material_list, scheme)  # Automatically choose C0 based on materials and scheme.
    elseif c0 isa Symbol
        c0 = chose_c0(material_list, c0)  # Choose C0 based on a specific  method.
    else
        c0 = c0  # Use the provided C0 reference material.
    end
    verbose_step ? (@info "c0" c0) : nothing

    phases_gpu = cu(phases)

    if precision == :double
        FT = Float64
    elseif precision == :simple
        FT = Float32
    else
        throw(ArgumentError("precision keyword must be :double or :simple (default :double)"))
    end

    material_list = [IE2ITE(m) for m in material_list]
    if T <: Complex
        FT = Complex{FT}
        material_list = [convert_to_complex_ITE(m) for m in material_list]
        P, Pinv, xi1, xi2, xi3, tau1, tau2, tau3, tau4, tau5, tau6 = init_gpu_complexfft(FT, size(phases)...)
    else
        P, Pinv, xi1, xi2, xi3, tau1, tau2, tau3, tau4, tau5, tau6 = init_gpu_realfft(FT, size(phases)...)
    end
    material_list_gpu = [m |> cu for m in material_list] |> cu

    eps1, eps2, eps3, eps4, eps5, eps6, sig1, sig2, sig3, sig4, sig5, sig6 = init_fields(FT, size(phases)...)

    EPS, SIG = zeros(FT, 6), zeros(FT, 6)

    step_hist = Hist2{FT}(length(loading_list))

    r = CUDA.zeros(FT, size(eps1))


    if scheme == FixedPoint
        step_solver! = fixed_point_step_solver_gpu!
    else
        step_solver! = polarization_step_solver_gpu!
    end

    keep_it_info ? fft_hist_list = [] : nothing

    if keep_fields
        epsf = zeros(FT, 6, size(phases)..., length(loading_list))
        sigf = zeros(FT, 6, size(phases)..., length(loading_list))
    end



    for loading_index in eachindex(loading_list)
        loading = loading_list[loading_index]

        keep_it_info ? (fft_hist = Hist2{FT}(Nit_max)) : (fft_hist = nothing)

        t = CUDA.@elapsed it, err_equi, err_comp, err_load = step_solver!(r, eps1, eps2, eps3, eps4, eps5, eps6, sig1, sig2, sig3, sig4, sig5, sig6, EPS, SIG, phases_gpu, material_list_gpu, tols, loading_type, loading, c0, P, Pinv, xi1, xi2, xi3, tau1, tau2, tau3, tau4, tau5, tau6, Nit_max, verbose_fft, fft_hist)


        verbose_step ? println("step time $t") : nothing

        verbose_step ? print_iteration(it, EPS, SIG, err_equi, err_comp, err_load, tols) : nothing
        if it == Nit_max
            @error "MAX ITERATIONS REACHED"
            return nothing
        end

        keep_it_info ? (push!(fft_hist_list, fft_hist)) : nothing

        ES = sum([1.0, 1.0, 1.0, 2.0, 2.0, 2.0] .* EPS .* SIG)
        update_hist!(step_hist, loading_index, E=EPS, S=SIG, ES=ES, equi=err_equi, comp=err_comp, load=err_load, it=it)


        if keep_fields
            epsf[:, :, :, :, loading_index] = zeros(FT, length(loading_list), 6, size(phases)...)
            sigf[:, :, :, :, loading_index] = zeros(FT, length(loading_list), 6, size(phases)...)
        end

        if save_fields
            #todo save to vtk or any
            @info "save_fields option not implemented yet"
        end


    end

    output = Dict(
        :steps => step_hist,
        :eps => keep_fields ? epsf : nothing,
        :sig => keep_fields ? sigf : nothing,
        :it_info => keep_it_info ? fft_hist_list : nothing,
    )

    return output
end



function fixed_point_step_solver_gpu!(r, eps1, eps2, eps3, eps4, eps5, eps6, sig1, sig2, sig3, sig4, sig5, sig6, EPS, SIG, phases_gpu, material_list_gpu, tols, loading_type, loading, c0, P, Pinv, xi1, xi2, xi3, tau1, tau2, tau3, tau4, tau5, tau6, Nit_max_green_operator_inv, verbose_fft, fft_hist)

    if loading_type == Strain

        CUDA.@. eps1 += loading[1] - EPS[1]
        CUDA.@. eps2 += loading[2] - EPS[2]
        CUDA.@. eps3 += loading[3] - EPS[3]
        CUDA.@. eps4 += loading[4] - EPS[4]
        CUDA.@. eps5 += loading[5] - EPS[5]
        CUDA.@. eps6 += loading[6] - EPS[6]
        n_blocks, n_threads = get_blocks_threads(eps1)
        @cuda blocks = n_blocks threads = n_threads rdcgpu!(sig1, sig2, sig3, sig4, sig5, sig6, eps1, eps2, eps3, eps4, eps5, eps6, phases_gpu, material_list_gpu)

        tol_equi = tols[1]
        err_equi = 1e9
        it = 0

        while err_equi > tol_equi && it < Nit_max_green_operator_inv
            it += 1

            gamma0!(P, Pinv, xi1, xi2, xi3, tau1, tau2, tau3, tau4, tau5, tau6, sig1, sig2, sig3, sig4, sig5, sig6, c0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            update_strain!(eps1, eps2, eps3, eps4, eps5, eps6, sig1, sig2, sig3, sig4, sig5, sig6)

            err_equi = Float64(eq_error(r, sig1, sig2, sig3, sig4, sig5, sig6))

            @cuda blocks = n_blocks threads = n_threads rdcgpu!(sig1, sig2, sig3, sig4, sig5, sig6, eps1, eps2, eps3, eps4, eps5, eps6, phases_gpu, material_list_gpu)

            EPS .= means_gpu(eps1, eps2, eps3, eps4, eps5, eps6)
            SIG .= means_gpu(sig1, sig2, sig3, sig4, sig5, sig6)

            isnothing(fft_hist) ? nothing : (update_hist!(fft_hist, it, E=EPS, S=SIG, equi=err_equi,))

            verbose_fft ? print_iteration(it, EPS, SIG, err_equi, 0.0, 0.0, tols) : nothing
        end
        if isnan(err_equi)
            throw(ErrorException("err_equi is NaN – check algorithm stability or parameter choices (e.g., c0)."))
        else
            return it, err_equi, 0.0, 0.0
        end
    elseif loading_type == Stress

        CUDA.@. sig1 = 0.0
        CUDA.@. sig2 = 0.0
        CUDA.@. sig3 = 0.0
        CUDA.@. sig4 = 0.0
        CUDA.@. sig5 = 0.0
        CUDA.@. sig6 = 0.0
        n_blocks, n_threads = get_blocks_threads(eps1)
        @cuda blocks = n_blocks threads = n_threads rdcinvgpu!(eps1, eps2, eps3, eps4, eps5, eps6, sig1, sig2, sig3, sig4, sig5, sig6, phases_gpu, material_list_gpu)

        tol_equi = tols[1]
        tol_load = tols[3]
        err_equi = 1e9
        err_load = 1e9
        it = 0

        SIG .= 0.0
        while (err_equi > tol_equi || err_load > tol_load) && it < Nit_max_green_operator_inv
            it += 1

            new_mean_eps = compute_eps(loading - SIG, c0)

            # gamma0!(tau, sig, c0, fftinfo, mean_value = new_mean_eps)
            gamma0!(P, Pinv, xi1, xi2, xi3, tau1, tau2, tau3, tau4, tau5, tau6, sig1, sig2, sig3, sig4, sig5, sig6, c0, new_mean_eps)

            update_strain!(eps1, eps2, eps3, eps4, eps5, eps6, sig1, sig2, sig3, sig4, sig5, sig6)

            err_equi = Float64(eq_error(r, sig1, sig2, sig3, sig4, sig5, sig6))

            @cuda blocks = n_blocks threads = n_threads rdcgpu!(sig1, sig2, sig3, sig4, sig5, sig6, eps1, eps2, eps3, eps4, eps5, eps6, phases_gpu, material_list_gpu)

            EPS .= means_gpu(eps1, eps2, eps3, eps4, eps5, eps6)
            SIG .= means_gpu(sig1, sig2, sig3, sig4, sig5, sig6)

            err_load = Float64(abs(sum((SIG .- loading) .^ 2 .* [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])))

            isnothing(fft_hist) ? nothing : (update_hist!(fft_hist, it, E=EPS, S=SIG, equi=err_equi, load=err_load))
            verbose_fft ? print_iteration(it, EPS, SIG, err_equi, 0.0, err_load, tols) : nothing
        end
        if isnan(err_equi)
            throw(ErrorException("err_equi is NaN – check algorithm stability or parameter choices (e.g., c0)."))
        else
            return it, err_equi, 0.0, err_load
        end
    end
end

