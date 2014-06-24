function nll(m::NystromModel)
	if !same_hyp(m)
		recompute_kernel_eig!(m)
	end
	refresh_cache!(m)
	sigma2 = exp(2m.lik.log_sigma)
	neig = length(m.nystrom.values)
	LambdaInv = diagm(neig./m.nystrom.values)
	Z = (m.PhiTPhi + sigma2 * LambdaInv)
	yTQy = (m.yTy - dot(m.PhiTy, Z \ m.PhiTy)) / sigma2
	logdetZ = logdet(Z)
	n = size(m.X, 1)
	scale = n - length(m.nystrom.values)
	logdetQ = scale*log(sigma2) + logdetZ + sum(log(m.nystrom.values))
	yTQy / 2 + logdetQ / 2 + scale*log(2pi)
end

function nll(m::FullModel)
	refresh_cache!(m)
	n = size(m.X, 1)
	dot(m.y, m.CinvY) / 2 + logdet(m.C) / 2 + n*log(2pi)
end

function nll(m::FITCModel)
	sigma2 = exp(2m.lik.log_sigma)
	Kmn = kernel(m.k, m.u, m.X)
	Km = kernel(m.k, m.u)
	Knn = kernel(m.k, m.X, m.X)
	Lambda = diagm(diag(Knn - Kmn' * (Km \ Kmn)))
	C = Kmn' * (Km \ Kmn) + Lambda + sigma2 * eye(Lambda)
	n = size(m.y)[1]
	(dot(m.y,(C \ m.y))  .+ logdet(C) .+ n*log(2pi))/2
end


# function nll(m::LaplacianModel)
# 	refresh_cache!(m)
# 	n = size(m.X, 1)
# 	sigma2 = exp(2m.lik.log_sigma)
# 	scale = n - length(m.lambdas)
# 	Ssqrtlambda = Float64[S(m.k, sqrt(lambda)) for lambda in m.lambdas]
# 	LambdaInv = diagm(1./Ssqrtlambda)
# 	Z = (m.PhiTPhi + sigma2 * LambdaInv)
# 	logdetZ = logdet(Z)
# 	logdetQ = scale*log(sigma2) + logdetZ + sum(log(Ssqrtlambda))
# 	yTQy = (m.yTy - dot(m.PhiTy, Z \ m.PhiTy)) / sigma2
# 	yTQy / 2 + logdetQ / 2 + scale*log(2pi)
# end


include("negloglik_laplacian.jl")