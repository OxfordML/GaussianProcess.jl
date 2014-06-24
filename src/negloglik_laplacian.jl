immutable SharedData{T<:FloatingPoint}
	Lambda::Diagonal{T}
	sigma2::T
	diagZ::Array{T, 1}
	Z::Union(Base.LinAlg.Cholesky{T}, Array{T,2})
end
function get_shared(m, k, lik)
	Lambda = Diagonal(Float64[S(k, sqrt(lam)) for lam in m.lambdas])
	sigma2 = exp(2lik.log_sigma)
	diagZ = sigma2 ./ diag(Lambda) + diag(m.PhiTPhi)
	Z = sigma2 * inv(Lambda) + m.PhiTPhi
	try
		Z = cholfact(Z)
	catch 
	end
	return SharedData(Lambda, sigma2, diagZ, Z)
end

function nll(m::LaplacianModel{OneD}) 
	s = get_shared(m, m.k, m.lik)
	nll(m, m.k, m.lik, s)
end

function nll(m, k, lik, s)
	n = size(m.X, 1)
	yTQy(m, k, s) / 2 + logdetQ(m, k, s) / 2 + n*log(2pi)
end

function dnlldl(m, k, lik)
	s = get_shared(m, k, lik)
	dnlldl(m, k, lik, s)
end
function dnlldnu(m, k, lik)
	s = get_shared(m, k, lik)
	dnlldnu(m, k, lik, s)
end

function dnlldsigma2(m, k, lik)
	s = get_shared(m, k, lik)
	dnlldsigma2(m, k, lik, s)
end

function dnlldl(m, k, lik, s)
	dyTQydl(m, k, s) / 2 + dlogdetQdl(m, k, s) / 2
end
function dnlldnu(m, k, lik, s)
	dyTQydnu(m, k, s) / 2 + dlogdetQdnu(m, k, s) / 2
end

function dnlldsigma2(m, k, lik, s)
	dyTQydsigma2(m, k, s) / 2 + dlogdetQdsigma2(m, k, s) / 2
end

function yTQy(m, k, s)
	(dot(m.y,m.y) - dot(m.PhiTy, s.Z \ m.PhiTy)) / s.sigma2
end

function dyTQydl(m, k, s)
	dLambdadl = Diagonal([dSdl(k, sqrt(lam)) for lam in m.lambdas])
	- dot(m.PhiTy, s.Z \ (inv(s.Lambda)^2 * dLambdadl *  (s.Z \ m.PhiTy)))
end

function dyTQydnu(m, k, s)
	dLambdadnu = Diagonal([dSdnu(k, sqrt(lam)) for lam in m.lambdas])
	- dot(m.PhiTy, s.Z \ (inv(s.Lambda)^2 * dLambdadnu *  (s.Z \ m.PhiTy)))
end

function dyTQydsigma2(m, k, s)
	2s.sigma2 * (dot(m.PhiTy, s.Z \ (inv(s.Lambda) * (s.Z \ m.PhiTy))) - yTQy(m, k, s)) / s.sigma2
end

function logdetQ(m, k, s)
	logdetZ = logdet(s.Z)
	n = size(m.X, 1)
	(n - m.jmax)*log(s.sigma2) + logdetZ + sum(log(diag(s.Lambda)))
end

function dlogdetQdl(m, k, s)
	dLambdadl = Diagonal([dSdl(k, sqrt(l)) for l in m.lambdas])
	trace(dLambdadl / s.Lambda) - s.sigma2 * sum(1 ./ (s.diagZ  .* inv(inv(s.Lambda)^2 * dLambdadl).diag))
end

function dlogdetQdnu(m, k, s)
	dLambdadnu = Diagonal([dSdnu(k, sqrt(l)) for l in m.lambdas])
	trace(dLambdadnu / s.Lambda) - s.sigma2 * sum(1 ./ (s.diagZ  .* inv(inv(s.Lambda)^2 * dLambdadnu).diag))
end

function dlogdetQdsigma2(m, k, s)
	n = size(m.X, 1)
	((n - m.jmax) / (s.sigma2) + sum(1./ (s.diagZ .* diag(s.Lambda)))) * 2s.sigma2
end