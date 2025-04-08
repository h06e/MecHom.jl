using Printf


struct Hist2{T}
	E::Array{T,2}
	S::Array{T,2}
	ES::Vector{T}
	equi::Vector{Float64}
	comp::Vector{Float64}
	load::Vector{Float64}
	it::Vector{Int64}

	function Hist2{T}(n::Int64) where T<:Number
		new(zeros(T,6,n), zeros(T,6,n), zeros(T,n), zeros(Float64,n), zeros(Float64,n), zeros(Float64,n), zeros(Int64,n))
	end
end

function update_hist!(
	hist::Hist2,
	i::Int64;
	E::Union{Nothing, Vector}=nothing,
	S::Union{Nothing, Vector}=nothing,
	ES::Union{Nothing, Number}=nothing,
	equi::Union{Nothing, Float64}=nothing,
	comp::Union{Nothing, Float64}=nothing,
	load::Union{Nothing, Float64}=nothing,
	it::Union{Nothing, Int64}=nothing
)
	if ! isnothing(E)
		hist.E[:,i] .= E
	end
	if ! isnothing(S)
		hist.S[:,i] .= S
	end
	if ! isnothing(ES)
		hist.ES[i] = ES
	end
	if ! isnothing(equi)
		hist.equi[i] = equi
	end
	if ! isnothing(comp)
		hist.comp[i] = comp
	end
	if ! isnothing(load)
		hist.load[i] = load
	end
	if ! isnothing(load)
		hist.load[i] = load
	end
	if ! isnothing(it)
		hist.it[i] = it
	end
end





struct Hist{T<:Union{Float64, ComplexF64}}
	E::Vector{Vector{T}}
	S::Vector{Vector{T}}
	ES::Vector{T}
	equi::Vector{Union{Nothing, Float64}}
	comp::Vector{Union{Nothing, Float64}}
	load::Vector{Union{Nothing, Float64}}
	eps::Vector{Union{Nothing, Array}}
	sig::Vector{Union{Nothing, Array}}
	ite::Vector{Union{Nothing, Int64}}

	function Hist{T}() where T<:Union{Float64,ComplexF64}
		new([], [], [], [], [], [], [], [], [])
	end
end

function update_hist!(
	hist::Hist,
	E::Vector{T},
	S::Vector{T};
	equi::Union{Nothing, Float64}=nothing,
	comp::Union{Nothing, Float64}=nothing,
	load::Union{Nothing, Float64}=nothing,
	eps::Union{Nothing, Array{T}}=nothing,
	sig::Union{Nothing, Array{T}}=nothing,
	ite::Union{Nothing, Int64}=nothing
) where T<:Union{Float64,ComplexF64}
	push!(hist.E, E)
	push!(hist.S, S)
	push!(hist.equi, equi)
	push!(hist.comp, comp)
	push!(hist.load, load)
	isnothing(eps) ? nothing : push!(hist.eps, eps)
	isnothing(sig) ? nothing : push!(hist.sig, sig)
	push!(hist.ite, ite)
end



function print_iteration(it::Int64, E::Vector{<:Union{Float64,Float32}}, S::Vector{<:Union{Float64,Float32}}, equi::Float64, comp::Float64, load::Float64,tols::Vector{Float64})
	printstyled("It: $it ")

	s = @sprintf("EQ:% 1.1e ", equi)
	equi[end] > tols[1] ? c = :red : c = :green
	printstyled(s, color = c, reverse = true)

	s = @sprintf("CO:% 1.1e ", comp)
	comp[end] > tols[2] ? c = :red : c = :green
	printstyled(s, color = c, reverse = true)

	s = @sprintf("LO:% 1.1e ", load)
	load[end] > tols[3] ? c = :red : c = :green
	printstyled(s, color = c, reverse = true)

	s = @sprintf("| E11:% 1.5e E22:% 1.5e E33:% 1.5e E23:% 1.5e E13:% 1.5e E12:% 1.5e |",
		E...)
	printstyled(s, color = :light_blue, reverse = true)

	s = @sprintf(" S11:% 1.5e S22:% 1.5e S33:% 1.5e S23:% 1.5e S13:% 1.5e S12:% 1.5e\n",
		S...)
	printstyled(s, color = :light_blue, reverse = false)
end

function print_iteration(it::Int64, E::Vector{<:Union{ComplexF64,ComplexF32}}, S::Vector{<:Union{ComplexF64,ComplexF32}}, equi::Float64, comp::Float64, load::Float64,tols::Vector{Float64})
	printstyled("It: $it ")

	s = @sprintf("EQ:% 1.1e ", equi)
	equi[end] > tols[1] ? c = :red : c = :green
	printstyled(s, color = c, reverse = true)

	s = @sprintf("CO:% 1.1e ", comp)
	comp[end] > tols[2] ? c = :red : c = :green
	printstyled(s, color = c, reverse = true)

	s = @sprintf("LO:% 1.1e ", load)
	load[end] > tols[3] ? c = :red : c = :green
	printstyled(s, color = c, reverse = true)

	s = @sprintf("| E11:% 1.5e + % 1.5ei E22:% 1.5e + % 1.5ei E33:% 1.5e + % 1.5ei E23:% 1.5e + % 1.5ei E13:% 1.5e + % 1.5ei E12:% 1.5e + % 1.5ei |",
	collect(Iterators.flatten(zip(real.(E), imag.(E))))...)
	printstyled(s, color = :light_blue, reverse = true)

	s = @sprintf(" S11:% 1.5e + % 1.5ei S22:% 1.5e + % 1.5ei S33:% 1.5e + % 1.5ei S23:% 1.5e + % 1.5ei S13:% 1.5e + % 1.5ei S12:% 1.5e + % 1.5ei\n",
	collect(Iterators.flatten(zip(real.(S), imag.(S))))...)
	printstyled(s, color = :light_blue, reverse = false)
end