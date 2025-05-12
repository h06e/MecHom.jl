using MecHom


function main()

    matrix = IE(E=4.0, nu=0.4)
    # fibre = IE(E=80.0, nu=0.2)
    fibre = ITE(El=230.0, Et=15.0, nul=0.2, mul=15.0, mut=10.0)
    material_list = [matrix, fibre]

    nf=200
    f=0.5
    dmin=0.1
    Np=1024

    info, micro = MecHom.Micro.gen_2d_random_disks(nf, f, dmin, Np, seed=123)
    # micro = rand((1,2),Np,Np,Np)

    loading_list = [[1.0,0.,0.,0.,0.,0.]]
    time_list = [Float64(i) for i in eachindex(loading_list)]

    tols = [1e-20,0.0, 1e-4]

   


    simple_out = solverGPU(micro, material_list, Strain, loading_list, time_list, tols; precision=:simple, keep_fields=false, verbose_step=true, verbose_fft=true,green_willot=false, c0=:ITE2DStrain, Nit_max=50)

 
end

main()