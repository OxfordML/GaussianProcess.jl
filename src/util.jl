import Base.length
function length(g::Union(Likelihood, Kernel))
	sum([length(g.(n)) for n in names(g)])
end