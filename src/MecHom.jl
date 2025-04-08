module MecHom

using Revise

include("microstructure.jl")
using .Micro

include("types.jl")

include("behaviors.jl")

include("choose_c0.jl")

include("fftinfo.jl")

include("gamma0.jl")

include("crit.jl")

include("history.jl")

include("solver_cpu.jl")

include("solver_gpu.jl")

end
