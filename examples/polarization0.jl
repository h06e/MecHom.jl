using MecHom

function main()
    # mat1 = IE(E=4.0, nu=0.4)
    mat1 = IE(E=4., nu=0.4)
    mat2 = IE(E=80.0, nu=0.2)
    # mat3 = IE2ITE(mat2)
    mat4 = ITE(El=230.0, Et=15.0, nul=0.3, mul=10.0, nut=0.07)
    # mat4 = ITE(El=Complex(230.0), Et=Complex(15.0), nul=Complex(0.3), mul=Complex(10.0), nut=Complex(0.07))

    # N1, N2, N3 = 128, 128, 128
    # micro = generate_micro(N1, N2, N3)

    info, micro = MecHom.Micro.gen_2d_random_disks(30, 0.5, 0.1, 1024; seed=123)

    loading_list = [[1.0,0.,0.,0.,0.,0.],[0.0,0.,0.,1.,0.,0.]]
    time_list = [Float64(i) for i in eachindex(loading_list)]

    tols = [1e-7, 1e-6, 1e-4]


    material_list = [mat1, mat4]
    # material_list = [IE(10.0, 5.0), IE(2.0, 1.0)]

    # solcpu = solver(
    #     micro,
    #     material_list,
    #     Strain,
    #     loading_list,
    #     time_list,
    #     tols;
    #     verbose_fft=true,
    #     verbose_step=true,
    #     # Nit_max_green_operator_inv=50,
    # )

    solgpu = solverGPU(
        micro,
        material_list,
        Strain,
        loading_list,
        time_list,
        tols;
        verbose_fft=false,
        verbose_step=true,
    )

    solgpu = solverGPU(
        micro,
        material_list,
        Strain,
        loading_list,
        time_list,
        tols;
        verbose_fft=true,
        verbose_step=true,
        scheme=MecHom.Polarization
    )



    return
end


function test0()
    fibre = ITE(El=230.0, Et=15.0, mul=15.0, nul=0.2, mut=7.0)
    matrix0 = IE(E=4.0, nu=0.4)
    matrix = IE(kappa=matrix0.kappa, mu = matrix0.mu/100)

    material_list = [matrix, fibre]
    info, micro = MecHom.Micro.gen_2d_random_disks(50, 0.5, 0.1, 2048; seed=123)


    loading_list = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    time_list = [ComplexF64(i) for i in 1:6]
    tols = [1e-7, 0.0, 1e-4]

    simple_out = solverGPU(micro, material_list, Strain, loading_list, time_list, tols; precision=:simple, keep_fields=false, verbose_step=true, verbose_fft=true, scheme=MecHom.Polarization)


end


# main()
test0()