import Sobol.SobolSeq

type NystromModel <: GP
	k::Kernel
	lik::Likelihood
	ub::Array{Float64,1}
	lb::Array{Float64,1}
	d::Int64
	neig_samples::Int64
	eig_ratio::Float64
	eig_samples::Array{Float64, 2}
	nystrom::Base.LinAlg.Eigen{Float64, Float64}
	scaled_vecs::Array{Float64, 2}
	X::Array{Float64, 2}
	y::Array{Float64, 1}
	y_raw::Array{Float64, 1}
	Phi::Array{Float64, 2}
	lambdas::Array{Float64, 1}
	PhiTPhi::Array{Float64, 2}
	PhiTy::Array{Float64, 1}
	yTy::Float64
	prev_hyp::Array{Float64, 1}
	Z::Array{Float64 ,2}
	Zchol::Base.LinAlg.Cholesky{Float64}
	stale_cache::Bool
	stale_space::Bool
	neig::Integer
	mu::Float64
	std::Float64
end


function NystromModel(k::Kernel, lik::Likelihood, ub::Array{Float64, 1}, 
	lb::Array{Float64, 1}; neig_samples=100, eig_ratio=100, neig=500)
	NystromModel(k, lik, ub, lb, 0, neig_samples, convert(Float64, eig_ratio),
		Array(Float64, (0, 0)), eigfact(eye(1)), Array(Float64, (0,0)),
		Array(Float64, (0, 0)),
		Float64[], Float64[], Array(Float64, (0,0)), Float64[], 
		Array(Float64, (0,0)),
		Float64[], 0.0, Array(Float64, (0,)),
		Array(Float64, (0,0)), cholfact(eye(1)), false, true, neig,
		0.0, 1.0)
end

function predict(m::NystromModel, xp)
	# refresh_cache!(m)
	if m.stale_cache
		refresh_cache!(m)
	end
	if m.stale_space
		recompute_kernel_eig!(m)
	end
	sigma2 = exp(2m.lik.log_sigma)
	k_star = kernel(m.k, xp, m.eig_samples)
	n = size(m.nystrom.values, 1)
	phi_star = sqrt(n) * k_star * m.scaled_vecs
	mn = vec(phi_star * (m.Zchol \ m.PhiTy))
	d = size(m.X, 2)
	c = kernel(m.k, zeros(1,d), zeros(1,d))
	variance = sigma2 * vec(sum(phi_star' .* (m.Zchol \ phi_star'), 1))
	mn * (m.std + eps()) .+ m.mu, variance * (m.std + eps())^2
end
function observe!{T <: FloatingPoint}(m::NystromModel, X::Array{T}, y::Array{T})
	if size(m.X) == (0,0)
		m.d = size(X, 2)
		println("Using $(m.neig_samples) samples")
		m.X = X
		m.y_raw = y
	else
		m.X = vcat(m.X, X)
		m.y_raw = vcat(m.y_raw, y)
	end
	m.mu = mean(m.y_raw)
	m.std = std(m.y_raw)
	m.y = (m.y_raw .- m.mu) / (m.std + eps())
	m.stale_cache = true
	m.stale_space = true
end

function refresh_cache!(m::NystromModel)
	if !same_hyp(m)
		recompute_kernel_eig!(m)
	end
	k_data_eig = kernel(m.k, m.X, m.eig_samples)
	n = length(m.nystrom.values)
	m.Phi = sqrt(n) * k_data_eig * m.scaled_vecs 
	m.PhiTPhi = m.Phi'm.Phi
	m.PhiTy = m.Phi'*m.y
	m.yTy = dot(m.y,m.y)
	sigma2 = exp(2m.lik.log_sigma)
	LambdaInv = diagm(n./m.nystrom.values)
	m.Z = m.PhiTPhi + sigma2 * LambdaInv
	m.Zchol = cholfact(m.Z)
	m.stale_cache = false

end

# Resamples the input space to characterize the eigenspace of the kernel.
function resample_input_space!(m::NystromModel)
	lb = [minimum(m.X[:, i]) for i = 1:m.d]
	ub = [maximum(m.X[:, i]) for i = 1:m.d]
	lb = min(lb, m.lb)
	ub = max(ub,  m.ub)
	seq = SobolSeq(m.d, lb, ub)
	[next(seq) for i = 1:m.neig_samples]
	m.eig_samples = hcat([next(seq) for i = 1:m.neig_samples]...)'
end

function recompute_kernel_eig!(m::NystromModel)
	resample_input_space!(m)
	m.prev_hyp = flatten_hyp(m)
	K = Base.LinAlg.Symmetric(kernel(m.k, m.eig_samples))
	try 
		max_eig = eigmax(K.S)
		if max_eig == 0
			throw(DomainError())
		end
		n = size(m.eig_samples,1)
		# m.nystrom = eigfact(K, max_eig / m.eig_ratio, max_eig)
		if m.neig < n
			m.nystrom = eigfact(K, n-m.neig: n)
		else
			m.nystrom = eigfact(K)
		end
	catch e
		println(e)
		m.nystrom = eigfact(K)
	end
	if length(m.nystrom.vectors) == m.neig_samples
		ratio = maximum(m.nystrom.vectors) / minimum(m.nystrom.vectors)
		println("Warning: you are probably not getting enough eigenvalues!")
		println("Ratio: $ratio, want $(m.eig_ratio) ")
	end
	m.scaled_vecs = m.nystrom.vectors ./ real(m.nystrom.values)'
	m.stale_space = false
end

function same_hyp(m::NystromModel)
	m.prev_hyp == [m.k.(n) for n in names(m.k)]
end