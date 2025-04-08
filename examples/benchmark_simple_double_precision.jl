using MecHom
using PyPlot

function main()
    mat1 = IE(E=4.0, nu=0.4)
    mat2 = ITE(El=230.0, Et=15.0, nul=0.3, mul=10.0, nut=0.07)

    info, micro = MecHom.Micro.gen_2d_random_disks(50, 0.5, 0.1, 1024; seed=123)
    loading_list = [[0.0, 0.0, 0.0, 1.0, 0.0, 0],[0.0, 0.0, 0.0, 1.0, 0.0, 0]]
    precision = [1e-8, 1e-6, 1e-4]

    time_list = [Float64(i) for i in eachindex(loading_list)]
    material_list = [mat1, mat2]

    # double_out =  solverGPU(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true, verbose_fft=true, precision=:double, c0=:ITE2DStrain, keep_it_info=true, keep_fields=false)

    simple_out =  solverGPU(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true, verbose_fft=true, precision=:simple, c0=:ITE2DStrain, keep_it_info=true, keep_fields=true)
       

    # steps = double_out[:steps]

    # println([steps.E[:,i] for i in 1:size(steps.E,2)])
    # println([steps.S[:,i] for i in 1:size(steps.S,2)])
    # println(steps.it)
    # println(steps.equi)

    # it_info = double_out[:it_info]

    # s23_double = [double_out[:it_info][1].S[4,i] for i in 1:double_out[:steps].it[1]]
    # s23_simple = [simple_out[:it_info][1].S[4,i] for i in 1:simple_out[:steps].it[1]]

    # pygui(true)
    # plt.figure()
    # plt.plot(collect(1:length(s23_double)),s23_double.-s23_double[end],"x", label="double")
    # plt.plot(collect(1:length(s23_simple)),s23_simple.-s23_double[end],"+", label="simple")
    # plt.legend()
    # plt.loglog()
    # plt.show()
    
    precision = [1e-12, 1e-6, 1e-4]
    double_out =  solverGPU(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true, verbose_fft=false, precision=:simple, c0=:ITE2DStrain, keep_it_info=false, keep_fields=false)
    double_out =  solverGPU(micro, material_list, Strain, loading_list, time_list, precision; verbose_step=true, verbose_fft=false, precision=:double, c0=:ITE2DStrain, keep_it_info=false, keep_fields=false)

end


main()