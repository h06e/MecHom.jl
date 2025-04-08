
export solver


# **************************************************************************
# *   MAIN SOLVER
# **************************************************************************

function solver(
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
) where {T<:Union{Float64,ComplexF64}}


    #-------------------------------------------------------------------
    for mat in material_list
        if (mat isa ITE{ComplexF64} || mat isa IE{ComplexF64}) && T == Float64
            @error "Cannot have Complex material in material_list AND Real time_list --> convert time_list into Vector{ComplexF64}"
            throw(ArgumentError("Cannot have Complex material in material_list AND Real time_list --> convert time_list into Vector{ComplexF64}"))
        end
    end

    #-------------------------------------------------------------------
    flag_only_ie = true
    for mat in material_list
        if mat isa ITE
            flag_only_ie = false
        end
    end
    # material_list = [IE2ITE(mat) for mat in material_list]

    if isnothing(c0)
        c0 = chose_c0(material_list, scheme)  # Automatically choose C0 based on materials and scheme.
    elseif c0 isa Symbol
        c0 = chose_c0(material_list, c0)  # Choose C0 based on a specific  method.
    else
        c0 = c0  # Use the provided C0 reference material.
    end
    verbose_step ? (@info "c0" c0) : nothing

    material_list = [GE(mat) for mat in material_list]

    if scheme == FixedPoint
        step_solver! = fixed_point_step_solver!
    else
        step_solver! = polarization_step_solver!
    end


    step_hist = Hist2{T}(length(loading_list))

    eps = zeros(T, 6, size(phases)...)
    sig = zeros(T, 6, size(phases)...)

    fftinfo, tau = FFTInfo(eps)
    EPS, SIG = zeros(T, 6), zeros(T, 6)

    keep_it_info ? fft_hist_list = [] : nothing

    if keep_fields
        epsf = zeros(FT, 6, size(phases)..., length(loading_list))
        sigf = zeros(FT, 6, size(phases)..., length(loading_list))
    end

    for loading_index in eachindex(loading_list)
        loading = loading_list[loading_index]

        keep_it_info ? (fft_hist = Hist2{FT}(Nit_max)) : (fft_hist = nothing)

        t = @elapsed it, err_equi, err_comp, err_load = step_solver!(fftinfo, fft_hist, tau, eps, sig, EPS, SIG, phases, material_list, tols, loading_type, loading, c0, polarization_AB, Nit_max, verbose_fft)

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
            epsf[:, :, :, :, loading_index] .= eps
            sigf[:, :, :, :, loading_index] .= sig
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



# **************************************************************************
# *   FIXED POINT STEP SOLVER
# **************************************************************************


function fixed_point_step_solver!(fftinfo, fft_hist, tau, eps, sig, EPS, SIG, phases, material_list, tols, loading_type, loading, c0, polarization_AB, Nit_max_green_operator_inv, verbose_fft)

    if loading_type == Strain
        eps .= reshape(loading - EPS, 6, 1, 1, 1) .+ eps #todo pas bien
        rdc!(sig, eps, phases, material_list)


        tol_equi = tols[1]
        err_equi = 1e9
        it = 0


        while err_equi > tol_equi && it < Nit_max_green_operator_inv
            it += 1

            gamma0!(tau, sig, c0, fftinfo, mean_value=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            eps .+= sig

            err_equi = eq_err(sig)

            rdc!(sig, eps, phases, material_list)

            EPS .= mean_field(eps)
            SIG .= mean_field(sig)

            isnothing(fft_hist) ? nothing : (update_hist!(fft_hist, it, E=EPS, S=SIG, equi=err_equi,))

            verbose_fft ? print_iteration(it, EPS, SIG, err_equi, 0.0, 0.0, tols) : nothing
        end

        if isnan(err_equi)
            throw(ErrorException("err_equi is NaN – check algorithm stability or parameter choices (e.g., c0)."))
        else
            return it, err_equi, 0.0, 0.0
        end
    elseif loading_type == Stress
        sig .= 0.0
        # sig .= reshape(loading, 6, 1, 1, 1) #todo pas bien
        rdc_inv!(eps, sig, phases, material_list)

        tol_equi = tols[1]
        tol_load = tols[3]
        err_equi = 1e9
        err_load = 1e9
        it = 0
        SIG .= 0.0
        while (err_equi > tol_equi || err_load > tol_load || it < 5) && it < Nit_max_green_operator_inv
            it += 1

            new_mean_eps = compute_eps(loading - SIG, c0)

            gamma0!(tau, sig, c0, fftinfo, mean_value=new_mean_eps)

            eps .+= sig

            err_equi = eq_err(sig)

            rdc!(sig, eps, phases, material_list)

            EPS .= mean_field(eps)
            SIG .= mean_field(sig)


            err_load = abs(sum((SIG .- loading) .^ 2 .* [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]))

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



# **************************************************************************
# *   EYRE MILTON STEP SOLVER
# **************************************************************************


function polarization_step_solver!(fftinfo, fft_hist, tau, eps, sig, EPS, SIG, phases, material_list, tols, loading_type, loading, c0, polarization_AB, Nit_max_green_operator_inv, verbose_fft)




end