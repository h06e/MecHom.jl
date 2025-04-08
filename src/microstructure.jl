module Micro
using StaticArrays
using Images
import Random

export gen_2d_random_disks


struct Disk
    x::SVector{2,Float64}  # current position of the disk
    xp::SVector{2,Float64} # previous position of the disk
    r::Float64
end

function random_new_disk(initial_speed::Float64)
    x = SVector{2,Float64}(rand(), rand())
    xp = x + initial_speed * (rand(2) .- 0.5)
    r = 0.0
    return Disk(x, xp, r)
end

function update_inertia!(disk_list::Vector{Disk}, damping::Float64)
    for is in eachindex(disk_list)
        s = disk_list[is]
        u = s.x - s.xp
        x_new = s.x + u * (1 - damping)
        xp = s.x
        updated_disk = Disk(x_new, xp, s.r)
        disk_list[is] = updated_disk
    end
end

function update_periodicity!(disk_list::Vector{Disk})
    for is in eachindex(disk_list)
        s = disk_list[is]
        xx = s.x[1]
        if xx < 0
            xx_new = s.x[1] + 1
            xxp_new = s.xp[1] + 1
        elseif xx > 1
            xx_new = s.x[1] - 1
            xxp_new = s.xp[1] - 1
        else
            xx_new = s.x[1]
            xxp_new = s.xp[1]
        end

        xy = s.x[2]
        if xy < 0
            xy_new = s.x[2] + 1
            xyp_new = s.xp[2] + 1
        elseif xy > 1
            xy_new = s.x[2] - 1
            xyp_new = s.xp[2] - 1
        else
            xy_new = s.x[2]
            xyp_new = s.xp[2]
        end

        disk_list[is] = Disk(SVector{2,Float64}(xx_new, xy_new), SVector{2,Float64}(xxp_new, xyp_new), s.r)
    end
end

function update_contact!(disk_list::Vector{Disk}, stiffness::Float64)
    d_min_meas = 1e9
    for is1 in eachindex(disk_list)
        for is2 in eachindex(disk_list)
            if is1 != is2
                s1 = disk_list[is1]
                s2 = disk_list[is2]

                k1 = round(s1.x[1] - s2.x[1])
                k2 = round(s1.x[2] - s2.x[2])
                closestB2A_dist2 = (s1.x[1] - s2.x[1] - k1) * (s1.x[1] - s2.x[1] - k1) + (s1.x[2] - s2.x[2] - k2) * (s1.x[2] - s2.x[2] - k2)
                d = sqrt(closestB2A_dist2)
                d_min_meas = min(d_min_meas, d)
                if closestB2A_dist2 <= (2 * s1.r)^2
                    normal = SVector{2,Float64}(s2.x[1] + k1, s2.x[2] + k2) - s1.x
                    overlap = 2 * s1.r - d
                    disp = normal * stiffness / d * 0.5 * overlap
                    x_s1_new = s1.x - disp
                    x_s2_new = s2.x + disp

                    disk_list[is1] = Disk(x_s1_new, s1.xp, s1.r)
                    disk_list[is2] = Disk(x_s2_new, s2.xp, s2.r)
                end
            end
        end
    end
    return d_min_meas
end

function update_radius!(disk_list::Vector{Disk}, radius::Float64)
    for is in eachindex(disk_list)
        s = disk_list[is]
        disk_list[is] = Disk(s.x, s.xp, radius)
    end
end


function add_sphere!(img,cx,cy,r)
    Np = size(img,1)
    xm, xM = Int(floor(max(1,cx*Np-r*Np))), Int(ceil(min(Np,cx*Np+r*Np)))
    ym, yM = Int(floor(max(1,cy*Np-r*Np))), Int(ceil(min(Np,cy*Np+r*Np)))
    for i in xm:xM
        for j in ym:yM
            d2=(i/Np-cx)^2+(j/Np-cy)^2
            if d2 <= (r^2)
                @inbounds img[i,j] = 2
            end
        end
    end
    return img
end

function conv2array(disk_list::Vector{Disk}, Np::Int64)
    img = zeros(Np, Np)
    # x = range(0, 1, length=Np)  # X-coordinates
    # y = range(0, 1, length=Np)  # Y-coordinates
    # X, Y = [xi for xi in x, yi in y], [yi for xi in x, yi in y]  # Meshgrid

    for s in disk_list
        cx, cy = s.x
        
        img = add_sphere!(img,cx,cy,s.r)
        if cx < s.r
            if cy < s.r
                img = add_sphere!(img,cx+1,cy,s.r)
                img = add_sphere!(img,cx+1,cy+1,s.r)
                img = add_sphere!(img,cx,cy+1,s.r)
            elseif cy > 1 - s.r
                img = add_sphere!(img,cx+1,cy,s.r)
                img = add_sphere!(img,cx+1,cy-1,s.r)
                img = add_sphere!(img,cx,cy-1,s.r)
            else
                img = add_sphere!(img,cx+1,cy,s.r)
            end

        elseif cx > 1 - s.r
            if cy < s.r
                img = add_sphere!(img,cx-1,cy,s.r)
                img = add_sphere!(img,cx-1,cy+1,s.r)
                img = add_sphere!(img,cx,cy+1,s.r)
            elseif cy > 1 - s.r
                img = add_sphere!(img,cx-1,cy,s.r)
                img = add_sphere!(img,cx-1,cy-1,s.r)
                img = add_sphere!(img,cx,cy-1,s.r)
            else
                img = add_sphere!(img,cx-1,cy,s.r)
            end
        else
            if cy < s.r
                img = add_sphere!(img,cx,cy+1,s.r)
            elseif cy > 1 - s.r
                img = add_sphere!(img,cx,cy-1,s.r)
            end
        end


        # img .+= ((X .- cx) .^ 2 + (Y .- cy) .^ 2 .<= s.r^2)

        # if cx < s.r
        #     if cy < s.r
        #         img .+= ((X .- cx .- 1) .^ 2 + (Y .- cy) .^ 2 .<= s.r^2)
        #         img .+= ((X .- cx .- 1) .^ 2 + (Y .- cy .- 1) .^ 2 .<= s.r^2)
        #         img .+= ((X .- cx) .^ 2 + (Y .- cy .- 1) .^ 2 .<= s.r^2)
        #     elseif cy > 1 - s.r
        #         img .+= ((X .- cx .- 1) .^ 2 + (Y .- cy) .^ 2 .<= s.r^2)
        #         img .+= ((X .- cx .- 1) .^ 2 + (Y .- cy .+ 1) .^ 2 .<= s.r^2)
        #         img .+= ((X .- cx) .^ 2 + (Y .- cy .+ 1) .^ 2 .<= s.r^2)
        #     else
        #         img .+= ((X .- cx .- 1) .^ 2 + (Y .- cy) .^ 2 .<= s.r^2)
        #     end

        # elseif cx > 1 - s.r
        #     if cy < s.r
        #         img .+= ((X .- cx .+ 1) .^ 2 + (Y .- cy) .^ 2 .<= s.r^2)
        #         img .+= ((X .- cx .+ 1) .^ 2 + (Y .- cy .- 1) .^ 2 .<= s.r^2)
        #         img .+= ((X .- cx) .^ 2 + (Y .- cy .- 1) .^ 2 .<= s.r^2)
        #     elseif cy > 1 - s.r
        #         img .+= ((X .- cx .+ 1) .^ 2 + (Y .- cy) .^ 2 .<= s.r^2)
        #         img .+= ((X .- cx .+ 1) .^ 2 + (Y .- cy .+ 1) .^ 2 .<= s.r^2)
        #         img .+= ((X .- cx) .^ 2 + (Y .- cy .+ 1) .^ 2 .<= s.r^2)
        #     else
        #         img .+= ((X .- cx .+ 1) .^ 2 + (Y .- cy) .^ 2 .<= s.r^2)
        #     end
        # else
        #     if cy < s.r
        #         img .+= ((X .- cx) .^ 2 + (Y .- cy .- 1) .^ 2 .<= s.r^2)
        #     elseif cy > 1 - s.r
        #         img .+= ((X .- cx) .^ 2 + (Y .- cy .+ 1) .^ 2 .<= s.r^2)
        #     end
        # end

    end
    return img
end


function save_fig_state(disk_list::Vector{Disk}, Np=512::Int64; name="tmp.png"::String)
    img = conv2array(disk_list, Np)

    min_val, max_val = minimum(img), maximum(img)
    arr_norm = (img .- min_val) ./ max((max_val - min_val), 1)
    gimg = colorview(Gray, arr_norm)
    save(name, gimg)
    return img
end


function gen_2d_random_disks(nf::Int64, f::Float64, d_min::Float64, Np::Int64;
    damping=0.1::Float64,
    stiffness=1.0::Float64,
    initial_speed=0.2::Float64,
    growing_rate=50::Int64,
    it_max=200::Int64,
    tol=1e-4::Float64,
    verbose=false::Bool,
    saveallsteps=false::Bool,
    saveallsteps_pix=512::Int64,
    seed=nothing::Union{Nothing,Int64})

    isnothing(seed) ? (seed=rand(Int64)) : nothing
    Random.seed!(seed)
    
    if f < 0 || f > 0.9
        @error "f must be between 0 and 0.9"
        throw(ArgumentError)
    end

    if d_min < 0
        @error "d_min must be >= 0"
        throw(ArgumentError)
    end
    true_radius_radius = sqrt(f / (nf * π)) # pix in radius
    final_radius = true_radius_radius * (1 + d_min / 2)


    disk_list = [random_new_disk(initial_speed) for _ in 1:nf]

    i = 1
    for r2 in LinRange(0, final_radius^2, growing_rate) #sqared root growing
        update_radius!(disk_list, sqrt(r2))
        update_periodicity!(disk_list)
        d_min_meas = update_contact!(disk_list, stiffness)
        update_inertia!(disk_list, damping)
        verbose ? (@info "$i d_min_meas = $d_min_meas") : nothing
        saveallsteps ? save_fig_state(disk_list, saveallsteps_pix, name="$i.png") : nothing
        i += 1
    end

    d_min_meas = 0.0
    while d_min_meas < 2 * final_radius - tol && i < it_max
        update_periodicity!(disk_list)
        d_min_meas = update_contact!(disk_list, stiffness)
        update_inertia!(disk_list, damping)
        saveallsteps ? save_fig_state(disk_list, saveallsteps_pix, name="$i.png") : nothing
        verbose ? (@info "$i d_min_meas = $d_min_meas | dmin-tol =$(2*final_radius-tol)") : nothing
        i += 1
    end
    update_periodicity!(disk_list)

    update_radius!(disk_list, true_radius_radius)
    img = conv2array(disk_list, Np)
    img = ones(size(img)) .* (img .> 0)

    true_f = sum(img)./length(img)
    true_d_min = d_min_meas / true_radius_radius - 2

    # Modification to fit the FFT solver 
    img = reshape(img, size(img, 1), size(img, 2), 1)
    img .+= 1 # to match with material_list indexes
    img = map(Int32, img)

    info = Dict(
        :f =>true_f,
        :dmin => true_d_min,
        :Dfpix => Int(round(2*true_radius_radius*Np)),
    )

    return info, img
end


end