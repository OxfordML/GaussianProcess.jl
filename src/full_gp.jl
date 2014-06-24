type FullModel <: GP
	d::Int64
	k::Kernel
	lik::Likelihood
	X::Array{Float64, 2}
	y::Array{Float64, 1}
	y_raw::Array{Float64, 1}
	K::Array{Float64, 2} #Covariance matrix
	Cfull::Array{Float64, 2}
    C::Union(Array{Float64, 2}, Base.LinAlg.Cholesky{Float64}) #Cholesky decomp of C
    CinvY::Array{Float64, 1} # C^-1  * Y 
    mu::Float64
    std::Float64
end

function FullModel(k::Kernel, lik::Likelihood)
	FullModel(0, k, lik, Array(Float64, (0, 0)), Array(Float64, (0,)),
		Array(Float64, (0,)),
		Array(Float64, (0,0)), Array(Float64, (0, 0)), Array(Float64, (0,0)),
		Array(Float64, (0,)), 0.0, 1.0)
end

function observe!{T <: FloatingPoint}(m::FullModel, X::Array{T}, y::Array{T})
	if size(m.X) == (0,0)
		m.d = size(X, 2)
		m.X = X
		m.y_raw = y
	else
		@assert m.d == size(X, 2) "Ensure X has the same dimension as the model!"
		m.X = vcat(m.X, X)
		m.y_raw = vcat(m.y_raw, y)
	end
	m.mu = mean(m.y_raw)
	m.std = std(m.y_raw)
	m.y = (m.y_raw .- m.mu) / (m.std + eps())
	refresh_cache!(m)
end

function predict(m::FullModel, Xp)
	Kpd = kernel(m.k, Xp, m.X)
	d = size(Xp, 2)
	z = zeros(1, d)
	c = kernel(m.k, z, z)[1]
	mn = (Kpd * m.CinvY) 
	variance = (c .- sum(Kpd .* (m.C \ Kpd')', 2)) 
	mn * (m.std + eps()) .+ m.mu, variance * (m.std + eps())^2
end

function refresh_cache!(m::FullModel)
	m.K = kernel(m.k, m.X)
	sigma2 = exp(2m.lik.log_sigma)
	m.Cfull = m.K + eye(m.K) * sigma2
	try
		m.C = cholfact(m.Cfull)
	catch e
		m.C = m.Cfull
	end
	m.CinvY = vec(m.C \ m.y)
end