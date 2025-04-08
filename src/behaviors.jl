using Parameters
using Adapt

export Material
export Elastic


abstract type Material end
abstract type Elastic <: Material end

export IE
export ITE
export IE2ITE

#**************************************************************

struct IE{T<:Union{Float64,ComplexF64}} <: Elastic
    kappa::T
    mu::T
    E::T
    nu::T
    lambda::T
end
Adapt.@adapt_structure IE


function IE(kappa::T, mu::T) where {T<:Union{Float64,ComplexF64}}
    E = 9.0 * kappa * mu / (3.0 * kappa + mu)
    nu = E / 2.0 / mu - 1.0
    lambda = kappa - 2.0 / 3.0 * mu
    return IE{T}(kappa, mu, E, nu, lambda)
end

function IE(; kwargs...)
    if haskey(kwargs, :E) && haskey(kwargs, :nu)
        kappa = kwargs[:E] / 3.0 / (1 - 2 * kwargs[:nu])
        mu = kwargs[:E] / 2 / (1 + kwargs[:nu])
        return IE(kappa, mu)
    elseif haskey(kwargs, :E) && haskey(kwargs, :mu)
        kappa = kwargs[:E] * kwargs[:mu] / 3 / (3 * kwargs[:mu] - kwargs[:E])
        return IE(kappa, kwargs[:mu])
    elseif haskey(kwargs, :kappa) && haskey(kwargs, :mu)
        return IE(kwargs[:kappa], kwargs[:mu])
    elseif haskey(kwargs, :lambda) && haskey(kwargs, :mu)
        kappa = kwargs[:lambda] + 2.0 / 3.0 * kwargs[:mu]
        return IE(kappa, kwargs[:mu])
    else
        @error "Bad arguments for IE material definition"
        throw(ArgumentError)
    end
end


function eigvals_mat(mat::IE)
    return 3 * mat.kappa, 2 * mat.mu
end

function IE2ITE(mat::IE)
    ITE(; k=mat.lambda + mat.mu,
        l=mat.lambda,
        m=mat.mu,
        n=mat.lambda + 2 * mat.mu,
        p=mat.mu)
end


#**************************************************************

struct ITE{T<:Union{Float64,ComplexF64}} <: Elastic
    k::T
    l::T
    m::T
    n::T
    p::T
    El::T
    Et::T
    nul::T
    nut::T
    mul::T
    mut::T
end
Adapt.@adapt_structure ITE

function ITE(k::T, l::T, m::T, n::T, p::T) where {T<:Union{Float64,ComplexF64}}
    El = n - l * l / k
    Et = 4 * m * k * El / (m * n + k * El)
    nul = l / 2.0 / k
    nut = (k * El - m * n) / (k * El + m * n)
    mut = Et / 2.0 / (1 + nut)
    mul = p

    ITE{T}(k, l, m, n, p, El, Et, nul, nut, mul, mut)
end

function ITE(; kwargs...)
    if all(key -> haskey(kwargs, key), [:k, :l, :m, :n, :p])
        @unpack k, l, m, n, p = kwargs
        return ITE(k, l, m, n, p)

    elseif all(key -> haskey(kwargs, key), [:El, :Et, :nul, :nut, :mul])
        @unpack El, Et, nul, nut, mul = kwargs
        p = mul
        deno = (El - El * nut - 2 * Et * nul^2)
        n = El^2 * (1 - nut) / deno
        l = El * Et * nul / deno
        kpm = Et * (El - Et * nul^2) / (1 + nut) / deno
        kmm = Et * (El * nut + Et * nul^2) / (1 + nut) / deno
        k = 0.5 * (kpm + kmm)
        m = 0.5 * (kpm - kmm)
        return ITE(; k=k, l=l, m=m, n=n, p=p)

    elseif all(key -> haskey(kwargs, key), [:El, :Et, :nul, :mut, :mul])
        @unpack El, Et, nul, mut, mul = kwargs
        nut = Et / 2 / mut - 1
        return ITE(; El=El, Et=Et, nul=nul, nut=nut, mul=mul)
    else
        @error "Bad arguments for ILE material definition"
        throw(ArgumentError)
    end
end

function eigvals_mat(mat::ITE)
    v1 = 2 * mat.m
    v2 = 2 * mat.p
    i = mat.k + mat.n / 2.0
    j =
        0.5 *
        sqrt(4 * mat.k * mat.k - 4 * mat.k * mat.n + 8 * mat.l * mat.l + mat.n * mat.n)
    v3 = i + j
    v4 = i - j
    return v1, v2, v3, v4
end


function IE2ITE(mat::ITE)
    mat
end

function convert_to_complex_ITE(mat::ITE)
    return ITE(k=Complex(mat.k), l=Complex(mat.l), m=Complex(mat.m), n=Complex(mat.n), p=Complex(mat.p))
end
    


#************************************************************************************
export GE

struct GE{T<:Union{Float64,ComplexF64}} <: Elastic
    isoclass::Int64

    kappa::T
    mu::T
    E::T
    nu::T
    lambda::T

    k::T
    l::T
    m::T
    n::T
    p::T
    El::T
    Et::T
    nul::T
    nut::T
    mul::T
    mut::T
end


function GE(mat::IE{T}) where T
    return GE{T}(1, mat.kappa, mat.mu, mat.E, mat.nu, mat.lambda,
        NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN,)
end

function GE(mat::ITE{T}) where T
    return GE{T}(2, NaN, NaN, NaN, NaN, NaN,
        mat.k, mat.l, mat.m, mat.n, mat.p, mat.El, mat.Et, mat.nul, mat.nut, mat.mul, mat.mut)
end

#************************************************************************************
#************************************************************************************




# function rdc!(sig::Array{T,4}, eps::Array{T,4},
#     phases::Array{Int32,3}, material_list) where {T<:Union{ComplexF64,Float64}}
#     _, N1, N2, N3 = size(eps)
#     @inbounds begin
#         for k in 1:N3
#             for j in 1:N2
#                 for i in 1:N1
#                     mat = material_list[phases[i, j, k]]
#                     if mat isa IE

#                         tre = eps[1, i, j, k] + eps[2, i, j, k] + eps[3, i, j, k]

#                         sig[1, i, j, k] = tre * mat.lambda + eps[1, i, j, k] * 2 * mat.mu
#                         sig[2, i, j, k] = tre * mat.lambda + eps[2, i, j, k] * 2 * mat.mu
#                         sig[3, i, j, k] = tre * mat.lambda + eps[3, i, j, k] * 2 * mat.mu
#                         sig[4, i, j, k] = eps[4, i, j, k] * 2 * mat.mu
#                         sig[5, i, j, k] = eps[5, i, j, k] * 2 * mat.mu
#                         sig[6, i, j, k] = eps[6, i, j, k] * 2 * mat.mu
#                     elseif mat isa ITE

#                         sig[1, i, j, k] = (mat.k + mat.m) * eps[1, i, j, k] + (mat.k - mat.m) * eps[2, i, j, k] + mat.l * eps[3, i, j, k]
#                         sig[2, i, j, k] = (mat.k - mat.m) * eps[1, i, j, k] + (mat.k + mat.m) * eps[2, i, j, k] + mat.l * eps[3, i, j, k]
#                         sig[3, i, j, k] = mat.l * eps[1, i, j, k] + mat.l * eps[2, i, j, k] + mat.n * eps[3, i, j, k]
#                         sig[4, i, j, k] = 2 * mat.p * eps[4, i, j, k]
#                         sig[5, i, j, k] = 2 * mat.p * eps[5, i, j, k]
#                         sig[6, i, j, k] = 2 * mat.m * eps[6, i, j, k]

#                     else
#                         @error "material is either IE or ITE. Error code 1957."
#                     end
#                 end
#             end
#         end
#     end
# end



function rdc!(sig::Array{T,4}, eps::Array{T,4},
    phases::Array{Int32,3}, material_list::Vector{<:GE}) where {T<:Union{ComplexF64,Float64}}
    _, N1, N2, N3 = size(eps)
    @inbounds begin
        for k in 1:N3
            for j in 1:N2
                for i in 1:N1

                    mat = material_list[phases[i, j, k]]

                    if mat.isoclass == 1

                        tre = eps[1, i, j, k] + eps[2, i, j, k] + eps[3, i, j, k]

                        sig[1, i, j, k] = tre * mat.lambda + eps[1, i, j, k] * 2 * mat.mu
                        sig[2, i, j, k] = tre * mat.lambda + eps[2, i, j, k] * 2 * mat.mu
                        sig[3, i, j, k] = tre * mat.lambda + eps[3, i, j, k] * 2 * mat.mu
                        sig[4, i, j, k] = eps[4, i, j, k] * 2 * mat.mu
                        sig[5, i, j, k] = eps[5, i, j, k] * 2 * mat.mu
                        sig[6, i, j, k] = eps[6, i, j, k] * 2 * mat.mu

                    elseif mat.isoclass == 2

                        sig[1, i, j, k] = (mat.k + mat.m) * eps[1, i, j, k] + (mat.k - mat.m) * eps[2, i, j, k] + mat.l * eps[3, i, j, k]
                        sig[2, i, j, k] = (mat.k - mat.m) * eps[1, i, j, k] + (mat.k + mat.m) * eps[2, i, j, k] + mat.l * eps[3, i, j, k]
                        sig[3, i, j, k] = mat.l * eps[1, i, j, k] + mat.l * eps[2, i, j, k] + mat.n * eps[3, i, j, k]
                        sig[4, i, j, k] = 2 * mat.p * eps[4, i, j, k]
                        sig[5, i, j, k] = 2 * mat.p * eps[5, i, j, k]
                        sig[6, i, j, k] = 2 * mat.m * eps[6, i, j, k]

                    else
                        @error "material is either IE or ITE. Error code 1957."
                    end
                end
            end
        end
    end
end



function rdc_inv!(eps::Array{T,4}, sig::Array{T,4},
    phases::Array{Int32,3}, material_list::Vector{<:GE}) where {T<:Union{ComplexF64,Float64}}

    _, N1, N2, N3 = size(eps)
    @inbounds begin
        for k in 1:N3
            for j in 1:N2
                for i in 1:N1

                    mat = material_list[phases[i, j, k]]

                    if mat.isoclass == 1

                        trs = sig[1] + sig[2] + sig[3]
                        unpnusE = (1 + mat.nu) / (mat.E)
                        mnusE = -mat.nu / mat.E
                        eps[1, i, j, k] = unpnusE * sig[1, i, j, k] + mnusE * trs
                        eps[2, i, j, k] = unpnusE * sig[2, i, j, k] + mnusE * trs
                        eps[3, i, j, k] = unpnusE * sig[3, i, j, k] + mnusE * trs
                        eps[4, i, j, k] = unpnusE * sig[4, i, j, k]
                        eps[5, i, j, k] = unpnusE * sig[5, i, j, k]
                        eps[6, i, j, k] = unpnusE * sig[6, i, j, k]

                    elseif mat.isoclass == 2

                        s11 = 1 / mat.Et
                        s33 = 1 / mat.El
                        s12 = -mat.nut / mat.Et
                        s13 = -mat.nul / mat.El
                        s44 = 1.0 / 2.0 / mat.mul
                        s66 = 1.0 / 2.0 / mat.mut

                        eps[1, i, j, k] = s11 * sig[1, i, j, k] + s12 * sig[2, i, j, k] + s13 * sig[3, i, j, k]
                        eps[2, i, j, k] = s12 * sig[1, i, j, k] + s11 * sig[2, i, j, k] + s13 * sig[3, i, j, k]
                        eps[3, i, j, k] = s13 * sig[1, i, j, k] + s13 * sig[2, i, j, k] + s33 * sig[3, i, j, k]
                        eps[4, i, j, k] = s44 * sig[4, i, j, k]
                        eps[5, i, j, k] = s44 * sig[5, i, j, k]
                        eps[6, i, j, k] = s66 * sig[6, i, j, k]

                    else
                        @error "material is either IE or ITE. Error code 1957."
                    end
                end
            end
        end
    end
end




function compute_eps(sig::Vector{T}, mat::IE) where {T<:Union{Float64,ComplexF64}}
    trs = sig[1] + sig[2] + sig[3]
    unpnusE = (1 + mat.nu) / (mat.E)
    mnusE = -mat.nu / mat.E
    eps = [
        unpnusE * sig[1] + mnusE * trs,
        unpnusE * sig[2] + mnusE * trs,
        unpnusE * sig[3] + mnusE * trs,
        unpnusE * sig[4],
        unpnusE * sig[5],
        unpnusE * sig[6],
    ]
    return eps
end

function compute_eps(sig::Vector{T}, mat::ITE) where {T<:Union{Float64,ComplexF64}}
    s11 = 1 / mat.Et
    s33 = 1 / mat.El
    s12 = -mat.nut / mat.Et
    s13 = -mat.nul / mat.El
    s44 = 1.0 / 2.0 / mat.mul
    s66 = 1.0 / 2.0 / mat.mut

    eps = [
        s11 * sig[1] + s12 * sig[2] + s13 * sig[3],
        s12 * sig[1] + s11 * sig[2] + s13 * sig[3],
        s13 * sig[1] + s13 * sig[2] + s33 * sig[3],
        s44 * sig[4],
        s44 * sig[5],
        s66 * sig[6],
    ]
    return eps
end



#**********************************************************************************
#* GPU
#**********************************************************************************


function rdcgpu!(sig1, sig2, sig3, sig4, sig5, sig6, eps1, eps2, eps3, eps4, eps5, eps6, phases, material_list)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if i <= length(eps1)
        mat = material_list[phases[i]]

        sig1[i] = (mat.k + mat.m) * eps1[i] + (mat.k - mat.m) * eps2[i] + mat.l * eps3[i]
        sig2[i] = (mat.k - mat.m) * eps1[i] + (mat.k + mat.m) * eps2[i] + mat.l * eps3[i]
        sig3[i] = mat.l * eps1[i] + mat.l * eps2[i] + mat.n * eps3[i]
        sig4[i] = 2 * mat.p * eps4[i]
        sig5[i] = 2 * mat.p * eps5[i]
        sig6[i] = 2 * mat.m * eps6[i]

    end
    return nothing
end


function rdcinvgpu!(eps1, eps2, eps3, eps4, eps5, eps6, sig1, sig2, sig3, sig4, sig5, sig6, phases, material_list)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if i <= length(eps1)
        mat = material_list[phases[i]]
        s11 = 1 / mat.Et
        s33 = 1 / mat.El
        s12 = -mat.nut / mat.Et
        s13 = -mat.nul / mat.El
        s44 = 1.0 / 2.0 / mat.mul
        s66 = 1.0 / 2.0 / mat.mut

        eps1[i] = s11 * sig1[i] + s12 * sig2[i] + s13 * sig3[i]
        eps2[i] = s12 * sig1[i] + s11 * sig2[i] + s13 * sig3[i]
        eps3[i] = s13 * sig1[i] + s13 * sig2[i] + s33 * sig3[i]
        eps4[i] = s44 * sig4[i]
        eps5[i] = s44 * sig5[i]
        eps6[i] = s66 * sig6[i]
    end
    return nothing
end