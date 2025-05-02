using MecHom


function main()


    matrix = IE(E=3.73, nu=0.38)
    fibre = ITE( El =  279.23600, Et =  35.52829, nul =  0.32791, nut =  0.07000, mul =  39.286614)
    material_list = [matrix, fibre, fibre]
    loading_list = [[0.,0.,1.,0.,0.,0.], ]
    time_list = [1.0]
    tols = [1e-7, 0.0, 1e-4]
    
    _, micro2D = MecHom.Micro.gen_2d_random_disks(8, 0.52, 0.1, 512; seed=123)
    data=Dict()

 
    c0=ITE(k=10.4818009685852, l=7.109684945740027, m=4.684105171128016, n=18.895349640392464,p= 4.13410186425809)

    simple_out = solverGPU(micro2D, material_list, Strain, loading_list, time_list, tols; precision=:simple, keep_fields=false, verbose_step=true, verbose_fft=false,green_willot=false, scheme=Polarization)

    simple_out = solverGPU(micro2D, material_list, Strain, loading_list, time_list, tols; precision=:simple, keep_fields=false, verbose_step=true, verbose_fft=false,green_willot=false, scheme=Polarization, c0 = c0)
    
    simple_out = solverGPU(micro2D, material_list, Stress, loading_list, time_list, tols; precision=:simple, keep_fields=false, verbose_step=true, verbose_fft=false,green_willot=false, scheme=Polarization)

    simple_out = solverGPU(micro2D, material_list, Stress, loading_list, time_list, tols; precision=:simple, keep_fields=false, verbose_step=true, verbose_fft=false,green_willot=false, scheme=Polarization, c0 = c0)
end

main()