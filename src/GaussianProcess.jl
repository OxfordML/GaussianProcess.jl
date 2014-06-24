module GaussianProcess
	include("coordinates.jl")
	include("likelihood.jl")
	include("cov.jl")
	include("spectrum.jl")
	include("util.jl")
	include("gp.jl")
	include("inference.jl")
	include("negloglik.jl")
	include("rand.jl")

	export SqExpIso,
		SqExpIsoPeriodic,
		SqExpIso2Periodic,
		MaternIso32,
		NystromModel,
		LaplacianModel,
		GaussianLikelihood,
		observe!,
		nll,
		nll_dnll!,
		infer!,
		MLE2,
		predict,
		CoordinateSystem,
		FullModel,
		Cartesian,
		MaternARD32,
		FITCModel,
		rand

end