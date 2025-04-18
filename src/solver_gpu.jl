using CUDA

export solverGPU




function eq_error!(r, sig, cartesian, nelmt)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    @inbounds if i <= nelmt

        i1 = cartesian[i][1]
        i2 = cartesian[i][2]
        i3 = cartesian[i][3]

        r[i] = sig[i1, i2, i3, 1] * sig[i1, i2, i3, 1] + sig[i1, i2, i3, 2] * sig[i1, i2, i3, 2] + sig[i1, i2, i3, 3] * sig[i1, i2, i3, 3] + 2 * sig[i1, i2, i3, 4] * sig[i1, i2, i3, 4] + 2 * sig[i1, i2, i3, 5] * sig[i1, i2, i3, 5] + 2 * sig[i1, i2, i3, 6] * sig[i1, i2, i3, 6]
        # r[i] = sig1[i]*sig1[i] + sig2[i]*sig2[i] + sig3[i]*sig3[i] + 2 * sig4[i]*sig4[i] + 2 * sig5[i]*sig5[i] + 2 * sig6[i]*sig6[i]
    end
    return nothing
end

function eq_error(r, sig, cartesian)
    nelmt = size(sig, 1) * size(sig, 2) * size(sig, 3)
    n_blocks, n_threads = get_blocks_threads(nelmt)
    @cuda blocks = n_blocks threads = n_threads eq_error!(r, sig, cartesian, nelmt)
    residu = reduce(+, r)
    residu = residu / nelmt
end


function meanfield(x)

    Nlength = size(x, 1) * size(x, 2) * size(x, 3)

    sums = CUDA.sum(reshape(x, :, size(x, 4)), dims=1)
    X = vec(sums) ./ Nlength
    return Array(X)
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
    precision::Symbol=:double,
    p::Vector{Float64}=[1.0, 1.0, 1.0],
    green_willot::Bool=false,
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

    freq_mod = nothing
    if green_willot
        freq_mod = modified_frequencied(FT, p, phases)
    end

    material_list = [IE2ITE(m) for m in material_list]
    if T <: Complex
        FT = Complex{FT}
        eps = CUDA.zeros(FT, size(phases)..., 6)
        sig = CUDA.zeros(FT, size(phases)..., 6)

        material_list = [convert_to_complex_ITE(m) for m in material_list]
        P, Pinv, xi1, xi2, xi3, tau = init_gpu_complexfft(FT,eps, p, phases)
    else
        eps = CUDA.zeros(FT, size(phases)..., 6)
        sig = CUDA.zeros(FT, size(phases)..., 6)

        P, Pinv, xi1, xi2, xi3, tau = init_gpu_realfft(FT,eps, p, phases)
    end

    cartesian = CartesianIndices(size(phases))

    material_list_gpu = [m |> cu for m in material_list] |> cu


    EPS = meanfield(eps)
    SIG = meanfield(sig)

    step_hist = Hist2{FT}(length(loading_list))

    r = CUDA.zeros(FT, size(phases))

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

        t = CUDA.@elapsed it, err_equi, err_comp, err_load = step_solver!(r, eps, sig, EPS, SIG, phases_gpu, material_list_gpu, tols, loading_type, loading, c0, P, Pinv, xi1, xi2, xi3, tau, Nit_max, verbose_fft, fft_hist, material_list, FT, freq_mod, cartesian,loading_index)


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
            epsf[:, :, :, :, loading_index] .= permutedims(Array(eps), (4, 1, 2, 3))
            sigf[:, :, :, :, loading_index] .= permutedims(Array(sig), (4, 1, 2, 3))

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



function fixed_point_step_solver_gpu!(r, eps, sig, EPS, SIG, phases_gpu, material_list_gpu, tols, loading_type, loading, c0, P, Pinv, xi1, xi2, xi3, tau, Nit_max, verbose_fft, fft_hist, material_list, FT, freq_mod, cartesian,loading_index)

    NNN = size(eps, 1) * size(eps, 2) * size(eps, 3)
    if loading_type == Strain

        add_mean_value!(eps, loading .- EPS, cartesian)

        n_blocks, n_threads = get_blocks_threads(NNN)
        @cuda blocks = n_blocks threads = n_threads rdcgpu!(sig, eps, phases_gpu, material_list_gpu, cartesian, NNN)

        tol_equi = tols[1]
        err_equi = 1e9
        it = 0

        while err_equi > tol_equi && it < Nit_max
            it += 1

            gamma0!(P, Pinv, xi1, xi2, xi3, tau, sig, c0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], freq_mod)

            CUDA.@. eps .+= sig

            err_equi = Float64(eq_error(r, sig, cartesian))

            @cuda blocks = n_blocks threads = n_threads rdcgpu!(sig, eps, phases_gpu, material_list_gpu, cartesian, NNN)

     
            EPS .= meanfield(eps)
            SIG .= meanfield(sig)
    
            isnothing(fft_hist) ? nothing : (update_hist!(fft_hist, it, E=EPS, S=SIG, equi=err_equi,))

            verbose_fft ? print_iteration(it, EPS, SIG, err_equi, 0.0, 0.0, tols) : nothing
        end


        if isnan(err_equi)
            throw(ErrorException("err_equi is NaN – check algorithm stability or parameter choices (e.g., c0)."))
        else
            return it, err_equi, 0.0, 0.0
        end
    elseif loading_type == Stress
        
        n_blocks, n_threads = get_blocks_threads(NNN)

        if loading_index ==1
            new_mean_eps = compute_eps(loading, c0)
            add_mean_value!(eps, new_mean_eps, cartesian)
            @cuda blocks = n_blocks threads = n_threads rdcgpu!(sig, eps, phases_gpu, material_list_gpu, cartesian, NNN)

            EPS .= meanfield(eps)
            SIG .= meanfield(sig)
        end

        tol_equi = tols[1]
        tol_load = tols[3]
        err_equi = 1e9
        err_load = 1e9
        it = 0

        SIG .= 0.0
        while (err_equi > tol_equi || err_load > tol_load) && it < Nit_max
            it += 1

            new_mean_eps = compute_eps(loading - SIG, c0)


            # gamma0!(tau, sig, c0, fftinfo, mean_value = new_mean_eps)
            gamma0!(P, Pinv, xi1, xi2, xi3, tau, sig, c0, new_mean_eps, freq_mod)

            CUDA.@. eps .+= sig

            err_equi = Float64(eq_error(r, sig, cartesian))

            @cuda blocks = n_blocks threads = n_threads rdcgpu!(sig, eps, phases_gpu, material_list_gpu, cartesian, NNN)

            EPS .= meanfield(eps)
            SIG .= meanfield(sig)

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





function polarization_step_solver_gpu!(r, eps, sig, EPS, SIG, phases_gpu, material_list_gpu, tols, loading_type, loading, c0, P, Pinv, xi1, xi2, xi3, tau, Nit_max, verbose_fft, fft_hist, material_list, FT, freq_mod, cartesian,loading_index)

    NNN = size(eps, 1) * size(eps, 2) * size(eps, 3)

    if loading_type == Strain

        alpha = 2.0
        beta = 2.0

        c0_list = [IE2ITE(c0) |> cu for m in material_list] |> cu
        cpc0 = [(m + c0) |> cu for m in material_list] |> cu

        sa = CUDA.zeros(FT, size(eps)...)
        sb = CUDA.zeros(FT, size(eps)...)
        
        add_mean_value!(eps, loading .- EPS, cartesian)

        n_blocks, n_threads = get_blocks_threads(NNN)
        @cuda blocks = n_blocks threads = n_threads rdcgpu!(sig, eps, phases_gpu, material_list_gpu, cartesian, NNN)

        tol_equi = tols[1]
        tol_load = tols[3]
        err_equi = 1e9
        err_load = 1e9
        it = 0

        while (err_equi > tol_equi || err_load > tol_load) && it < Nit_max
            it += 1

            @cuda blocks = n_blocks threads = n_threads rdcgpu!(sa, eps, phases_gpu, c0_list,cartesian, NNN)


            CUDA.@. sb = alpha * sig - beta * sa
            CUDA.@. sa = sig + (1 - beta) * sa

            gamma0!(P, Pinv, xi1, xi2, xi3, tau, sb, c0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], freq_mod)

            add_mean_value!(sb, beta * loading, cartesian)
  
            @cuda blocks = n_blocks threads = n_threads rdcgpu!(sig, sb, phases_gpu, c0_list,cartesian, NNN)

            CUDA.@. sig += sa
 
            @cuda blocks = n_blocks threads = n_threads rdcinvgpu!(sa, sig, phases_gpu, cpc0,cartesian, NNN)

            CUDA.@. sb = eps - sa

            err_equi = Float64(eq_error(r, sb, cartesian))

            CUDA.@. eps = sa

            @cuda blocks = n_blocks threads = n_threads rdcgpu!(sig, eps, phases_gpu, material_list_gpu,cartesian, NNN)

            EPS .= meanfield(eps)
            SIG .= meanfield(sig)


            err_load = Float64(abs(sum((EPS .- loading) .^ 2 .* [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])))


            isnothing(fft_hist) ? nothing : (update_hist!(fft_hist, it, E=EPS, S=SIG, equi=err_equi, load=err_load))
            verbose_fft ? print_iteration(it, EPS, SIG, err_equi, 0.0, err_load, tols) : nothing
        end
        if isnan(err_equi)
            throw(ErrorException("err_equi is NaN – check algorithm stability or parameter choices (e.g., c0)."))
        else
            return it, err_equi, 0.0, err_load
        end
    elseif loading_type == Stress
        @error "Polarization & Stress control is not implemented in MecHom yet."
        return
    end
end

