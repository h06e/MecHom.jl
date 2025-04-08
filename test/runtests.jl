using MecHom
using Test
using CUDA

function generate_micro(N1, N2, N3)
    phases = ones(Int32, N1, N2, N3)
    r2 = (N1 / 4)^2
    @inbounds begin
        for k in 1:N3
            for j in 1:N2
                for i in 1:N1
                    if ((i - N1 / 2)^2 + (j - N2 / 2)^2 + (k - N3 / 3)^2) < r2
                        phases[i, j, k] = 2
                    end
                end
            end
        end
    end
    return phases
end


@testset "CPU FixedPoint & Real" begin
    mat1 = IE(E=4.0, nu=0.4)
    mat2 = IE(E=80.0, nu=0.2)
    mat3 = IE2ITE(mat2)
    mat4 = ITE(El=230.0, Et=15.0, nul=0.3, mul=10.0, nut=0.07)

    N1, N2, N3 = 32, 32, 1
    micro = generate_micro(N1, N2, N3)
    # info, micro = MecHom.Micro.gen_2d_random_disks(1, 0.5, 0.1, 32; seed=123)
    loading_list = [[1.0, 0.0, 0.0, 0.0, 0.0, 0], [0.0, 1.0, 0.0, 0.0, 0.0, 0]]
    precision = [1e-6, 1e-6, 1e-4]

    time_list = [Float64(i) for i in eachindex(loading_list)]


    material_list = [mat1, mat2]
    @test solver(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true) !== nothing
    @test solver(micro, material_list, Stress, loading_list, time_list, precision; verbose_step=true) !== nothing

    material_list = [mat1, mat3]
    @test solver(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true) !== nothing
    @test solver(micro, material_list, Stress, loading_list, time_list, precision; verbose_step=true) !== nothing

    material_list = [mat1, mat4]
    @test solver(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true) !== nothing
    @test solver(micro, material_list, Stress, loading_list, time_list, precision; verbose_step=true) !== nothing

    material_list = [mat1, mat4]
    @test solver(micro, material_list, Strain, loading_list, time_list, precision; c0=:ITE2DStrain, verbose_step=true) !== nothing


end





@testset "CPU FixedPoint & Complex" begin
    mat1 = IE(E=4.0 + 1.0im, nu=0.4)
    mat2 = IE(E=80.0, nu=0.2)
    mat3 = IE2ITE(mat2)
    mat4 = ITE(El=230.0, Et=15.0, nul=0.3, mul=10.0, nut=0.07)

    N1, N2, N3 = 32, 32, 1
    micro = generate_micro(N1, N2, N3)
    # info, micro = MecHom.Micro.gen_2d_random_disks(1, 0.5, 0.1, 32; seed=123)
    loading_list = [[1.0, 0.0, 0.0, 0.0, 0.0, 0], [0.0, 1.0, 0.0, 0.0, 0.0, 0]]
    precision = [1e-6, 1e-6, 1e-4]

    time_list = [ComplexF64(i) for i in eachindex(loading_list)]


    material_list = [mat1, mat2]
    @test solver(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true) !== nothing
    @test solver(micro, material_list, Stress, loading_list, time_list, precision; verbose_step=true) !== nothing

    material_list = [mat1, mat3]
    @test solver(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true) !== nothing
    @test solver(micro, material_list, Stress, loading_list, time_list, precision; verbose_step=true) !== nothing

    material_list = [mat1, mat4]
    @test solver(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true) !== nothing
    @test solver(micro, material_list, Stress, loading_list, time_list, precision; verbose_step=true) !== nothing

    material_list = [mat1, mat4]
    @test solver(micro, material_list, Strain, loading_list, time_list, precision; c0=:ITE2DStrain, verbose_step=true) !== nothing

end



if CUDA.has_cuda() && CUDA.functional()
@testset "GPU  FixedPoint & Real" begin
    mat1 = IE(E=4.0, nu=0.4)
    mat2 = IE(E=80.0, nu=0.2)
    mat3 = IE2ITE(mat2)
    mat4 = ITE(El=230.0, Et=15.0, nul=0.3, mul=10.0, nut=0.07)

    N1, N2, N3 = 32, 32, 1
    micro = generate_micro(N1, N2, N3)
    # info, micro = MecHom.Micro.gen_2d_random_disks(1, 0.5, 0.1, 32; seed=123)
    loading_list = [[1.0, 0.0, 0.0, 0.0, 0.0, 0], [0.0, 1.0, 0.0, 0.0, 0.0, 0]]
    precision = [1e-6, 1e-6, 1e-4]

    time_list = [Float64(i) for i in eachindex(loading_list)]


    material_list = [mat1, mat2]
    @test solverGPU(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true) !== nothing
    @test solverGPU(micro, material_list, Stress, loading_list, time_list, precision; verbose_step=true) !== nothing

    material_list = [mat1, mat3]
    @test solverGPU(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true) !== nothing
    @test solverGPU(micro, material_list, Stress, loading_list, time_list, precision; verbose_step=true) !== nothing

    material_list = [mat1, mat4]
    @test solverGPU(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true) !== nothing
    @test solverGPU(micro, material_list, Stress, loading_list, time_list, precision; verbose_step=true) !== nothing

    material_list = [mat1, mat4]
    @test solverGPU(micro, material_list, Strain, loading_list, time_list, precision; c0=:ITE2DStrain, verbose_step=true) !== nothing

end




@testset "GPU  FixedPoint & Complex" begin
    mat1 = IE(E=4.0+1.0im, nu=0.4)
    mat2 = IE(E=80.0, nu=0.2)
    mat3 = IE2ITE(mat2)
    mat4 = ITE(El=230.0, Et=15.0, nul=0.3, mul=10.0, nut=0.07)

    N1, N2, N3 = 32, 32, 1
    micro = generate_micro(N1, N2, N3)
    # info, micro = MecHom.Micro.gen_2d_random_disks(1, 0.5, 0.1, 32; seed=123)
    loading_list = [[1.0, 0.0, 0.0, 0.0, 0.0, 0], [0.0, 1.0, 0.0, 0.0, 0.0, 0]]
    precision = [1e-6, 1e-6, 1e-4]

    time_list = [ComplexF64(i) for i in eachindex(loading_list)]


    material_list = [mat1, mat2]
    @test solverGPU(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true) !== nothing
    @test solverGPU(micro, material_list, Stress, loading_list, time_list, precision; verbose_step=true) !== nothing

    material_list = [mat1, mat3]
    @test solverGPU(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true) !== nothing
    @test solverGPU(micro, material_list, Stress, loading_list, time_list, precision; verbose_step=true) !== nothing

    material_list = [mat1, mat4]
    @test solverGPU(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true) !== nothing
    @test solverGPU(micro, material_list, Stress, loading_list, time_list, precision; verbose_step=true) !== nothing

    material_list = [mat1, mat4]
    @test solverGPU(micro, material_list, Strain, loading_list, time_list, precision; c0=:ITE2DStrain, verbose_step=true) !== nothing

end

else
    println("CUDA not available.")
end