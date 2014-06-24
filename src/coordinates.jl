abstract CoordinateSystem

type OneD <: CoordinateSystem
	L
end

type Cartesian <: CoordinateSystem
	dimensions
	limits
end

type TwoDPolar <: CoordinateSystem
	a
end

function λ(c::OneD, j::Integer)
	((j * π / (2c.L)) ^ 2)::Float64
end

function φ(c::OneD, x::Array{Float64, 1}, j::Integer)
	x = x[1]
	(sin(π * j * (x .+ c.L) / (2c.L)) / sqrt(c.L))::Float64
end

function λ(c::Cartesian, js::Array{Int64, 1})
	ans = 0
	for d = 1:c.dimensions
		d
		L = c.limits[d]
		ans += ((js[d] * π / (2L)) ^ 2)::Float64
	end
	ans
end

function φ(c::Cartesian, x::Array{Float64, 1}, js::Array{Int64, 1})
	ans = 1
	for d = 1:c.dimensions
		L = c.limits[d]
		ans *= (sin(π * js[d] * (x[d] .+ L) / (2L)) / sqrt(L))::Float64
	end
	ans
end
