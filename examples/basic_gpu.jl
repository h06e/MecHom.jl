using MecHom
using Profile
using PyPlot


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



function main()
    # mat1 = IE(E=4.0, nu=0.4)
    mat1 = IE(E=4.0+1.0im, nu=Complex(0.4))
    mat2 = IE(E=80.0, nu=0.2)
    mat3 = IE2ITE(mat2)
    mat4 = ITE(El=230.0, Et=15.0, nul=0.3, mul=10.0, nut=0.07)
    mat4 = ITE(El=Complex(230.0), Et=Complex(15.0), nul=Complex(0.3), mul=Complex(10.0), nut=Complex(0.07))

    # N1, N2, N3 = 128, 128, 128
    # micro = generate_micro(N1, N2, N3)

    info, micro = MecHom.Micro.gen_2d_random_disks(2, 0.5, 0.2, 128; seed=123)

    loading_list = [[1.0,0.,0.,0.,0.,0.]]
    time_list = [Float64(i) for i in eachindex(loading_list)]

    tols = [1e-6, 1e-6, 1e-4]

    # micro = zeros(Int32, 512,512,1) .+ Int32(2)
    # @info "" typeof(micro)
    material_list = [IE(E=3.5, nu=0.4), ITE(El=279.2, Et=33.1, nul=0.319, nut=0.07, mul=71.0)]
    # material_list = [IE(10000.0, 5000.0), IE(2.0, 1.0)]


    # solcpu = solver(
    #     micro,
    #     material_list,
    #     Strain,
    #     loading_list,
    #     time_list,
    #     tols;
    #     verbose_fft=false,
    #     verbose_step=true,
    #     # Nit_max_green_operator_inv=50,
    # )

    # solgpu = solverGPU(
    #     micro,
    #     material_list,
    #     Strain,
    #     loading_list,
    #     time_list,
    #     tols;
    #     verbose_fft=true,
    #     verbose_step=true,
    #     Nit_max=50,
    #     # c0=:ITE2DStrain
    # )

    # solgpu = solverGPU(
    #     micro,
    #     material_list,
    #     Strain,
    #     loading_list,
    #     time_list,
    #     tols;
    #     verbose_fft=false,
    #     verbose_step=true,
    #     Nit_max=500,
    #     precision=:simple,
    #     green_willot=false,
    #     # scheme=Polarization,
    #     keep_fields=true,
    #     keep_it_info=true
    # )

    matrix = material_list[1]
    fibre = material_list[2]

    @info "" fibre.k fibre.p fibre.m
    @info "" matrix.kappa matrix.mu

    @info "c0?" 0.5*(fibre.k+matrix.kappa) 0.5*(matrix.mu+fibre.p)

    solgpu2 = solverGPU(
        micro,
        material_list,
        Stress,
        loading_list,
        time_list,
        tols;
        verbose_fft=false,
        verbose_step=true,
        Nit_max=500,
        precision=:simple,
        green_willot=true,
        keep_fields=true,
        keep_it_info=true,
        c0=IE(kappa=100.0, mu=30.0)
    )

    # pygui(true)
    # plt.figure()
    # plt.plot(solgpu[:it_info][1].S[1,:])
    # plt.plot(solgpu2[:it_info][1].S[1,:])

    # pygui(true)
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(solgpu[:eps][1,:,:,:,1])
    # plt.subplot(122)
    # plt.imshow(solgpu2[:eps][1,:,:,:,1])
    # plt.show()

    # solver(
    #     micro,
    #     material_list,
    #     Stress,
    #     loading_list,
    #     time_list,
    #     precision;
    #     verbose_fft=false
    # )

    return
end





main()
# mainITE()
# mainITE2()
# maincomplex()