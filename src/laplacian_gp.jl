type LaplacianModel{C <: CoordinateSystem} <: GP
	k::Kernel
	lik::Likelihood
	coordinates::C
	jmax::Integer
	js::Array{Int64, 1}
	X::Array{Float64, 2}
	y::Array{Float64, 1}
	y_raw::Array{Float64, 1}
	stale::Bool
	Phi::Array{Float64, 2}
	lambdas::Array{Float64, 1}
	PhiTPhi::Array{Float64, 2}
	PhiTy::Array{Float64, 1}
	yTy::Float64
	Z::Array{Float64, 2}
	Zchol::Base.LinAlg.Cholesky{Float64}
	mu::Float64
	std::Float64
end

LaplacianModel{C <: CoordinateSystem}(k::Kernel, lik::Likelihood, c::C, jmax::Integer) = LaplacianModel(k, lik, c, jmax, Array(Int64, (0, )), 
		Array(Float64, (0, 0)), Array(Float64, (0, )), Array(Float64, (0, )),
		true,
		Array(Float64, (0, 0)), Array(Float64, (0, )), 
		Array(Float64, (0, 0)), Array(Float64, (0,)), 0.0,
		Array(Float64, (0, 0)), cholfact(eye(1)),
		0.0, 1.0)

function observe!{T <: FloatingPoint}(m::LaplacianModel, X::Array{T}, y::Array{T})
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
	refresh_cache!(m)
	m.stale = true
end

function predict(m::LaplacianModel, Xp::Array{Float64, 2})
	if m.stale
		refresh_cache!(m)
	end
	sigma2 = exp(2m.lik.log_sigma)
	Ssqrtλ = Float64[ S( m.k, √(λ) ) for λ in m.lambdas ]
	Λinv = eye(diagm(Ssqrtλ)) / diagm(Ssqrtλ)
	size(m.js, 1)
	phi_star = Float64[φ(m.coordinates, Xp[r, :][:], m.js[jr]) for r = 1:size(Xp, 1), jr = 1:size(m.js, 1)]
	Z = (m.PhiTPhi + sigma2 * Λinv)
	size(Z), size(phi_star), size(m.PhiTy)
	mu_p = vec(phi_star * (Z \ m.PhiTy)) * m.std .+ m.mu
	sigma_p = m.std^2 * vec(sigma2 * sum(phi_star' .* (Z \ phi_star'), 1))
	mu_p, sigma_p
end

###################################################################

function refresh_cache!(m::LaplacianModel)
	m.js = get_js(m)
	m.lambdas = [λ(m.coordinates,  m.js[jr]) for jr = 1:size(m.js, 1)]

	m.Phi = [φ(m.coordinates, m.X[r, :][:], m.js[jr]) for r = 1:size(m.X, 1), jr = 1:size(m.js, 1)]

	m.PhiTPhi = m.Phi'm.Phi
	m.PhiTy = m.Phi'*m.y
	m.yTy = dot(m.y,m.y)
	m.stale = false
end

function get_js{C <: Cartesian}(m::LaplacianModel{C})
	Int64[int(mod(floor(i/(m.jmax^(d-1))) , m.jmax)) for i = 0:((m.jmax^m.coordinates.dimensions)-1), d = 1:m.coordinates.dimensions] .+ 1
end

function get_js{C <: OneD}(m::LaplacianModel{C})
	[1:m.jmax]
end