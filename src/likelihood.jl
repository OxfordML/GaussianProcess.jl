abstract Likelihood

type GaussianLikelihood <: Likelihood
	log_sigma::Float64
end
GaussianLikelihood(v::Array{Float64,}) = GaussianLikelihood(v[1])

