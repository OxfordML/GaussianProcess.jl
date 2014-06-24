import Sobol.SobolSeq

type FITCModel <: GP
	k::Kernel
	lik::Likelihood
	u
	X
	y
	y_raw
	mu
	std
end


function FITCModel(k::Kernel, lik::Likelihood, u)
	FITCModel(k, lik, u, Array(Float64, (0,0)), Float64[], Float64[], 0, Inf)
end

function predict(m::FITCModel, Xp)
	sigma2 = exp(2m.lik.log_sigma)
	Kmn = kernel(m.k, m.u, m.X)
	Kss = kernel(m.k ,Xp)
	Km = kernel(m.k, m.u)
	Knn = kernel(m.k, m.X, m.X)
	Ks = kernel(m.k, Xp, m.u)
	Lambda = diagm(diag(Knn - Kmn' * (Km \ Kmn)))
	Qm = Km + Kmn * ((Lambda + sigma2 * eye(Lambda)) \ Kmn')
	mn = Ks * (Qm \ Kmn) * ((Lambda + sigma2 * eye(Lambda)) \ m.y)
	variance = diag(Kss - Ks * (inv(Km) - inv(Qm)) * Ks' .+ sigma2)
	mn * (m.std + eps()) .+ m.mu, variance * (m.std + eps())^2
end
function observe!{T <: FloatingPoint}(m::FITCModel, X::Array{T}, y::Array{T})
	if size(m.X) == (0,0)
		m.X = X
		m.y_raw = y
	else
		m.X = vcat(m.X, X)
		m.y_raw = vcat(m.y_raw, y)
	end
	m.mu = mean(m.y_raw)
	m.std = std(m.y_raw)
	m.y = (m.y_raw .- m.mu) / (m.std + eps())
end
