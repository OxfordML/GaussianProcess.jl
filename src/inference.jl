import NLopt

type MLE2 
reps
iterations
lb
ub
maxtime
maxeval
jitter
method
end
MLE2(lb, ub;reps=5, iterations=50, maxtime=10, maxeval=1000, jitter=1, method=:LN_COBYLA) = MLE2(reps, iterations, lb, ub, maxtime, maxeval, jitter, method)


type MLE2_DIRECT end



function wrap_model(m::GP)
	function wrapper_nll(hyp, grad)
		if any(isnan(hyp))
			return Inf
		end
		try
			m.k, m.lik = inflate_hyp(m, hyp)
			return nll(m)
		catch e
			return Inf
		end
	end
	return wrapper_nll
end

function infer!(o::MLE2, m::GP)
	init = flatten_hyp(m)
	init_best = (nll(m), init)
	best = (nll(m), init, :ORIGINAL)
	opt = NLopt.Opt(o.method, length(init))
	NLopt.lower_bounds!(opt, o.lb)
	NLopt.upper_bounds!(opt, o.ub)
	# NLopt.xtol_rel!(opt,1e-4)
	NLopt.ftol_rel!(opt, 1e-12)
	NLopt.maxtime!(opt, o.maxtime)
	NLopt.maxeval!(opt, o.maxeval)
	NLopt.min_objective!(opt, wrap_model(m))
	for r = 1:o.reps
		this_init = init + randn(size(init)) * o.jitter
		this_init = min(init, o.ub)
		this_init = max(init, o.lb)
		candidate = NLopt.optimize(opt, this_init)
		if candidate[1] < best[1]
			best = candidate
		end
	end
	println("Hyperparameter inf: $(best[3])")
	m.k, m.lik = inflate_hyp(m, best[2])
end

function infer!(o::MLE2, m::FITCModel)
	init = flatten_hyp(m)
	init_best = (nll(m), init)
	best = (nll(m), init, :ORIGINAL)
	opt = NLopt.Opt(o.method, length(init))
	NLopt.lower_bounds!(opt, o.lb[1:length(init)])
	NLopt.upper_bounds!(opt, o.ub[1:length(init)])
	NLopt.xtol_rel!(opt,1e-4)
	NLopt.ftol_rel!(opt, 1e-12)
	NLopt.maxtime!(opt, o.maxtime)
	NLopt.maxeval!(opt, o.maxeval)
	NLopt.min_objective!(opt, wrap_model(m))
	for r = 1:o.reps
		this_init = init + randn(size(init)) * o.jitter
		this_init = min(init, o.ub[1:length(init)])
		this_init = max(init, o.lb[1:length(init)])
		candidate = NLopt.optimize(opt, this_init)
		if candidate[1] < best[1]
			best = candidate
		end
	end
	m.k, m.lik = inflate_hyp(m, best[2][1:length(flatten_hyp(m))])
end
