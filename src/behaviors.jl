using Parameters
using CUDA
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


# function Base.:+(m3::ITE{T}, m4::ITE{T}) where T
#     ITE(k=m4.k + m3.k, l=m4.l + m3.l, m=m4.m + m3.m, n=m4.n + m3.n, p=m4.p + m3.p)
# end

function Base.:+(m1::Elastic, m2::Elastic)
    m3 = IE2ITE(m1)
    m4 = IE2ITE(m2)
    ITE(k=m4.k + m3.k, l=m4.l + m3.l, m=m4.m + m3.m, n=m4.n + m3.n, p=m4.p + m3.p)
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


function rdcgpu!(sig, eps, phases, material_list, cartesian, NNN)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if i <= NNN

        i1 = cartesian[i][1]
        i2 = cartesian[i][2]
        i3 = cartesian[i][3]

        mat = material_list[phases[i]]

        sig[i1,i2,i3,1] = (mat.k + mat.m) * eps[i1,i2,i3,1] + (mat.k - mat.m) * eps[i1,i2,i3,2] + mat.l * eps[i1,i2,i3,3]
        sig[i1,i2,i3,2] = (mat.k - mat.m) * eps[i1,i2,i3,1] + (mat.k + mat.m) * eps[i1,i2,i3,2] + mat.l * eps[i1,i2,i3,3]
        sig[i1,i2,i3,3] = mat.l * eps[i1,i2,i3,1] + mat.l * eps[i1,i2,i3,2] + mat.n * eps[i1,i2,i3,3]
        sig[i1,i2,i3,4] = 2 * mat.p * eps[i1,i2,i3,4]
        sig[i1,i2,i3,5] = 2 * mat.p * eps[i1,i2,i3,5]
        sig[i1,i2,i3,6] = 2 * mat.m * eps[i1,i2,i3,6]

    end
    return nothing
end


function rdcinvgpu!(eps, sig, phases, material_list, cartesian, NNN)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if i <= NNN
        mat = material_list[phases[i]]

        i1 = cartesian[i][1]
        i2 = cartesian[i][2]
        i3 = cartesian[i][3]

        s11 = 1 / mat.Et
        s33 = 1 / mat.El
        s12 = -mat.nut / mat.Et
        s13 = -mat.nul / mat.El
        s44 = 1.0 / 2.0 / mat.mul
        s66 = 1.0 / 2.0 / mat.mut

        eps[i1,i2,i3,1] = s11 * sig[i1,i2,i3,1] + s12 * sig[i1,i2,i3,2] + s13 * sig[i1,i2,i3,3]
        eps[i1,i2,i3,2] = s12 * sig[i1,i2,i3,1] + s11 * sig[i1,i2,i3,2] + s13 * sig[i1,i2,i3,3]
        eps[i1,i2,i3,3] = s13 * sig[i1,i2,i3,1] + s13 * sig[i1,i2,i3,2] + s33 * sig[i1,i2,i3,3]
        eps[i1,i2,i3,4] = s44 * sig[i1,i2,i3,4]
        eps[i1,i2,i3,5] = s44 * sig[i1,i2,i3,5]
        eps[i1,i2,i3,6] = s66 * sig[i1,i2,i3,6]
    end
    return nothing
end