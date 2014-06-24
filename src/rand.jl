import Distributions
import Base.rand

function rand(gp::FullModel, locations)
	Kpp = kernel(gp.k, locations, locations)
	Kdp = kernel(gp.k, gp.X, locations)
	K = Kpp - Kdp' * (gp.C \ Kdp)
	mean = predict(gp, locations)[1]
	rand(Distributions.MvNormal(mean, K))
end