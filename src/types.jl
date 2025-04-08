
#**************************************************************

export LoadingType
export Strain
export Stress

abstract type LoadingType end
abstract type Strain <: LoadingType end
abstract type Stress <: LoadingType end

#**************************************************************

export Scheme
export FixedPoint
export Polarization

abstract type Scheme end
abstract type FixedPoint <: Scheme end
abstract type Polarization <: Scheme end


#**************************************************************

export PU
export GPU
export CPU

abstract type PU end
abstract type GPU <: PU end
abstract type CPU <: PU end