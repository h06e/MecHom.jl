using MecHom

#* Microstructure parameters
nf = 30 # Number of fibres in the domain
f=0.5 # Volume fraction of fibre 
dmin=0.2 # Minimal distance between the fibres in fibre radius ratio.
N_pix=512 # Discretization of the microstructure into N_pix x N_pix

#* See gen_basic_rve documentation for more details (specify seed, growing rate...)
info, micro = MecHom.Micro.gen_2d_random_disks(nf, f, dmin, N_pix; seed=123)
