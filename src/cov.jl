abstract Covariance
import Distances

abstract Kernel

function flatten_hyp(m)
	lik_hyp = Float64[]
	for n in names(m.lik)
		push!(lik_hyp, m.lik.(n)...)	
	end
	cov_hyp = Float64[]
	for n in names(m.k)
		push!(cov_hyp, m.k.(n)...)
	end
	vcat(lik_hyp, cov_hyp)

end

function inflate_hyp(m, hyp_vec)
	liks = hyp_vec[1:length(m.lik)]
	covs = hyp_vec[length(m.lik) + 1:end]
	typeof(m.k)(covs), typeof(m.lik)(liks)
end

type SqExpIso <: Kernel
	log_l
	log_s
end
SqExpIso(v::Array) = SqExpIso(v[1], v[2])

function kernel{T <: FloatingPoint}(h::SqExpIso, X1::Array{T, 2}, 
						X2::Array{T, 2}) 
	l2 = exp(2h.log_l)
	s2 = exp(2h.log_s)
	s2 * exp(-Distances.pairwise(Distances.SqEuclidean(), X1', X2') / (2l2))
end
kernel{T <: FloatingPoint}(h::SqExpIso, X::Array{T, 2}) = kernel(h, X, X)

type SqExpIsoPeriodic <: Kernel
	log_l
	log_s
	log_period
end
SqExpIsoPeriodic(v::Array) = SqExpIsoPeriodic(v[1], v[2], v[3])

function kernel{T <: FloatingPoint}(h::SqExpIsoPeriodic, X1::Array{T, 2}, 
						X2::Array{T, 2}) 
	l2 = exp(2h.log_l)
	s2 = exp(2h.log_s)
	period1 = exp(h.log_period)
	s2 * 
	exp(-Distances.pairwise(Distances.SqEuclidean(), 
		cos(2pi * X1 / period1)', cos(2pi * X2 / period1)') / (2l2))
end
kernel{T <: FloatingPoint}(h::SqExpIsoPeriodic, X::Array{T, 2}) = kernel(h, X, X)

type SqExpIso2Periodic <: Kernel
	log_l
	log_s
	log_period1
	log_period2
end
SqExpIso2Periodic(v::Array) = SqExpIso2Periodic(v[1], v[2], v[3], v[4])

function kernel{T <: FloatingPoint}(h::SqExpIso2Periodic, X1::Array{T, 2}, 
						X2::Array{T, 2}) 
	l2 = exp(2h.log_l)
	s2 = exp(2h.log_s)
	period1 = exp(h.log_period1)
	period2 = exp(h.log_period2)
	s2 * 
	(exp(-Distances.pairwise(Distances.SqEuclidean(), 
		cos(2pi * X1 / period1)', cos(2pi * X2 / period1)') / (2l2)) .+
	exp(-Distances.pairwise(Distances.SqEuclidean(), 
		cos(2pi * X1 / period2)', cos(2pi * X2 / period2)') / (2l2)))
end
kernel{T <: FloatingPoint}(h::SqExpIso2Periodic, X::Array{T, 2}) = kernel(h, X, X)


type SqExpIso3Periodic <: Kernel
	log_l
	log_s
	log_period1
	log_period2
	log_period3
end
SqExpIso3Periodic(v::Array) = SqExpIso(v[1], v[2], v[3], v[4], v[5])

function kernel{T <: FloatingPoint}(h::SqExpIso3Periodic, X1::Array{T, 2}, 
						X2::Array{T, 2}) 
	l2 = exp(2h.log_l)
	s2 = exp(2h.log_s)
	period1 = exp(h.log_period1)
	period2 = exp(h.log_period2)
	period3 = exp(h.log_period3)
	s2 * 
	(exp(-Distances.pairwise(Distances.SqEuclidean(), 
		cos(2pi * X1 / period1)', cos(2pi * X2 / period1)') / (2l2)) .+
	exp(-Distances.pairwise(Distances.SqEuclidean(), 
		cos(2pi * X1 / period2)', cos(2pi * X2 / period2)') / (2l2)) .+ 
	exp(-Distances.pairwise(Distances.SqEuclidean(), 
		cos(2pi * X1 / period3)', cos(2pi * X2 / period3)') / (2l2)))
end
kernel{T <: FloatingPoint}(h::SqExpIso3Periodic, X::Array{T, 2}) = kernel(h, X, X)

type SqExpIso4Periodic <: Kernel
	log_l
	log_s
	log_period1
	log_period2
	log_period3
	log_period4
end
SqExpIso4Periodic(v::Array) = SqExpIso(v[1], v[2], v[3], v[4], 
	v[5], v[6], v[7])

function kernel{T <: FloatingPoint}(h::SqExpIso4Periodic, X1::Array{T, 2}, 
						X2::Array{T, 2}) 
	l2 = exp(2h.log_l)
	s2 = exp(2h.log_s)
	period1 = exp(h.log_period1)
	period2 = exp(h.log_period2)
	period3 = exp(h.log_period3)
	period4 = exp(h.log_period4)
	s2 * 
	(exp(-Distances.pairwise(Distances.SqEuclidean(), 
		cos(2pi * X1 / period1)', cos(2pi * X2 / period1)') / (2l2)) .+
	exp(-Distances.pairwise(Distances.SqEuclidean(), 
		cos(2pi * X1 / period2)', cos(2pi * X2 / period2)') / (2l2)) .+ 
	exp(-Distances.pairwise(Distances.SqEuclidean(), 
		cos(2pi * X1 / period3)', cos(2pi * X2 / period3)') / (2l2)) .+ 
	exp(-Distances.pairwise(Distances.SqEuclidean(), 
		cos(2pi * X1 / period4)', cos(2pi * X2 / period4)') / (2l2)))
end
kernel{T <: FloatingPoint}(h::SqExpIso4Periodic, X::Array{T, 2}) = kernel(h, X, X)


type MaternIso32 <: Kernel
	log_l
	log_s
end
MaternIso32(v::Array) = MaternIso32(v[1], v[2])
function kernel{T <: FloatingPoint}(h::MaternIso32, X1::Array{T, 2}, 
						X2::Array{T, 2})
	l = exp(h.log_l)
	s = exp(h.log_s)
	s3dol = sqrt(3) * Distances.pairwise(Distances.Euclidean(), X1', X2') / l
	s * (1 .+ s3dol) .* exp(-s3dol)
end
kernel{T <: FloatingPoint}(h::MaternIso32, X::Array{T, 2}) = kernel(h, X, X)

type MaternARD32 <: Kernel
	log_ls::Array{Float64,}
	log_s
end
MaternARD32(v::Array) = MaternARD32(v[1:end-1], v[end])
function kernel{T <: FloatingPoint}(h::MaternARD32, X1::Array{T, 2}, 
						X2::Array{T, 2})
	ls2 = 2exp(2h.log_ls[:])
	s = exp(h.log_s)
	s3dol = sqrt(3) * Distances.pairwise(Distances.WeightedEuclidean(1./ls2), X1', X2')
	s * (1 .+ s3dol) .* exp(-s3dol)
end
kernel{T <: FloatingPoint}(h::MaternARD32, X::Array{T, 2}) = kernel(h, X, X)

