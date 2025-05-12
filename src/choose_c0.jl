using Optim
using LinearAlgebra

export chose_c0


function normF(m::Matrix)
    return sum(m.^2)
end

function getmat(mat)
    m = MecHom.IE2ITE(mat)
    return [[m.k+m.m m.k-m.m m.l 0. 0. 0.];
    [m.k-m.m m.k+m.m m.l 0. 0. 0.];
    [m.l m.l m.n 0. 0. 0.];
    [0. 0. 0. 2*m.p 0. 0.];
    [0. 0. 0. 0. 2*m.p 0.];
    [0. 0. 0. 0. 0. 2*m.m]]

end

#! costfunc_c0 pour schema en polarisation
# function cost3(x,param)
#     C0 = getmat(ITE(k=x[1], l=x[2], m=x[3], n=x[4], p=x[5]))

#     s=0.
#     for i in [:matrix,:fibre]
#         C = getmat(param[i])
#         δC = C-C0
#         s= max(s,svd(inv(δC) * C).S[1])
#         s= max(s,svd(inv(δC) * C0).S[1])
#     end
#     return s
# end

function costfunc_c0(x,material_list)
    C0 = getmat(ITE(k=x[1], l=x[2], m=x[3], n=x[4], p=x[5]))

    s=0.
    for m in material_list
        C = getmat(m)
        δC = C-C0
        s= max(s,svd(inv(C0) * δC).S[1])
    end
    return s
end


function mat2x(mat)
    m = MecHom.IE2ITE(mat)
    return [m.k, m.l, m.m, m.n, m.p]
end


function chose_c0(material_list::Vector{<:Material},scheme::Type{<:Scheme})
    if scheme == FixedPoint

        # Chose c0 that minimize max_Ω ( C₀⁻¹δC )
        x0 = mat2x(material_list[1])
        result = optimize(x -> costfunc_c0(x, material_list), x0, NelderMead())
        x=Optim.minimizer(result)
        return ITE(k=x[1], l=x[2], m=x[3], n=x[4], p=x[5])


        if material_list isa Vector{<:IE}
            return chose_c0(material_list, :iso_arit_kappa_mu)
        elseif material_list isa Vector{<:Elastic}
            return chose_c0(material_list, :iso_arit_lambda0)
        end
    else
        if material_list isa Vector{<:IE}
            return chose_c0(material_list, :iso_geom_lambda0)
        elseif material_list isa Vector{<:Elastic}
            return chose_c0(material_list, :iso_geom_lambda0)
        end

    end
end


# ------------------------------------


function chose_c0(material_list::Vector{<:Material},method::Symbol)
    
    if method == :iso_arit_kappa_mu
        
        k_min, k_max = Inf, -Inf
        mu_min, mu_max = Inf, -Inf
        for material in material_list
            
            k_min = min(k_min, abs(material.kappa))
            k_max = max(k_max, abs(material.kappa))
            mu_min = min(mu_min, abs(material.mu))
            mu_max = max(mu_max, abs(material.mu))

        end
        return IE(kappa = 0.5 * (k_min + k_max), mu = 0.5 * (mu_min + mu_max))
    
    elseif method == :ITE2DStrain

        v1_min, v1_max = Inf, -Inf
        v2_min, v2_max = Inf, -Inf
        v3_min, v3_max = Inf, -Inf
        v4_min, v4_max = Inf, -Inf
        for material in material_list
            if material isa IE
                vps = eigvals_mat(IE2ITE(IE(kappa=abs(material.kappa),
                    mu = abs(material.mu))))
            else
                vps = eigvals_mat(IE2ITE(material))
            end
            v1_min = min(v1_min, abs(vps[1]))
            v1_max = max(v1_max, abs(vps[1]))
            v2_min = min(v2_min, abs(vps[2]))
            v2_max = max(v2_max, abs(vps[2]))
            v3_min = min(v3_min, abs(vps[3]))
            v3_max = max(v3_max, abs(vps[3]))
            v4_min = min(v4_min, abs(vps[4]))
            v4_max = max(v4_max, abs(vps[4]))
        end
        m0 = 0.5 * 0.5 * (v1_min + v1_max)
        p0 = 0.5 * 0.5 * (v2_min + v2_max)
        l0 = 0.
        k0 = 0.5 * 0.5 * (v3_min + v3_max)
        n0 = 0.5 * (v4_min + v4_max)

        return ITE(k = k0, l = l0, m = m0, n = n0, p = p0)

    elseif method == :iso_arit_lambda0
    
        eigval_min, eigval_max = Inf, -Inf
        for material in material_list
            vps = eigvals_mat(IE2ITE(material))

            eigval_min = min(eigval_min, minimum(abs.(vps)))
            eigval_max = max(eigval_max, maximum(abs.(vps)))

        end
        return IE(lambda = 0., mu = 0.5 * (eigval_min + eigval_max))

    elseif method == :iso_geom_lambda0
        
        eigval_min, eigval_max = Inf, -Inf
        for material in material_list
            vps = eigvals_mat(IE2ITE(material))

            eigval_min = min(eigval_min, minimum(abs.(vps)))
            eigval_max = max(eigval_max, maximum(abs.(vps)))

        end
        return IE(lambda = 0., mu = sqrt((eigval_min * eigval_max)) )
    else
        @error "No method called $method for chosing C0" 
        throw(ArgumentError)
    end

end
