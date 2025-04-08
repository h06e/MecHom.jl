using MecHom


function main()
    mat1 = IE(E=4.0, nu=0.4)
    mat2 = IE(E=80.0, nu=0.2)
    mat3 = IE2ITE(mat2)
    mat4 = ITE(El=230.0, Et=15.0, nul=0.3, mul=10.0, nut=0.07)



    info, micro = MecHom.Micro.gen_2d_random_disks(30, 0.5, 0.1, 128; seed=123)

    loading_list = [[1.0,0.,0.,0.,0.,.0]]
    time_list = [Float64(i) for i in eachindex(loading_list)]

    precision = [1e-6, 1e-6, 1e-4]


    material_list = [mat1, mat2]
    solver(
        micro,
        material_list,
        Strain,
        loading_list,
        time_list,
        precision;
        verbose_fft=false,
        verbose_step=false
    )

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


function mainITE()
    mat1 = IE(E=4.0, nu=0.4)
    mat2 = ITE(El=230.0, Et=15.0, nul=0.3, mul=10.0, nut=0.07)
    # mat2 = IE2ITE(IE(E=80.0, nu=0.2))
    material_list = [mat1, mat2]

    info, micro = MecHom.Micro.gen_2d_random_disks(30, 0.5, 0.1, 32; seed=123)

    loading_list = [[1.0,0.,0.,0.,0.,.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0]]
    # loading_list = [[0.0,1.,0.,0.,0.,.0]]
    time_list = [Float64(i) for i in eachindex(loading_list)]

    precision = [1e-6, 1e-6, 1e-4]

    solver(
        micro,
        material_list,
        Strain,
        loading_list,
        time_list,
        precision;
        c0=:ITE2DStrain,
        verbose_fft=false,
        verbose_step=true
    )


    return
end


function mainITE2()

    mat1 = IE(E=4.0, nu=0.4)
    mat2 = IE(E=80.0, nu=0.2)
    mat3 = IE2ITE(mat2)
    mat4 = ITE(El=230.0, Et=15.0, nul=0.3, mul=10.0, nut=0.07)


    info, micro = MecHom.Micro.gen_2d_random_disks(1, 0.5, 0.1, 32; seed=123)
    loading_list = [[1.0, 0.0, 0.0, 0.0, 0.0, 0], [0.0, 1.0, 0.0, 0.0, 0.0, 0]]
    precision = [1e-6, 1e-6, 1e-4]

    time_list = [Float64(i) for i in eachindex(loading_list)]

    material_list = [mat1, mat4]
    # solver(micro, material_list, Strain, loading_list, time_list, precision; c0=:ITE2DStrain, verbose_step=true) !== nothing
    # solver(micro, material_list, Stress, loading_list, time_list, precision; c0=:ITE2DStrain, verbose_fft=true, verbose_step=true) !== nothing


end




main()
# mainITE()
# mainITE2()
# maincomplex()