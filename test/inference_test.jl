reload("../src/GaussianProcess.jl")
module t
using GaussianProcess, PyPlot
d = 1
k = SqExpIso(-0.4, -0.7)
lik = GaussianLikelihood(-2.3)
nystrom = NystromModel(k, lik, 5, neig_samples=100)
full = FullModel(k, lik)
sarkka = LaplacianModel(k, lik, Cartesian(d, [5 for i = 1:d]), 16)

X = (rand(10, d) -0.5) * 6
y = sin(vec(sum(X,2))) .* exp(-vec(sum(X, 2)).^2 / 2) .+ cos(vec(sum(X,2)) * 2) .* exp(-vec(sum(X, 2)).^2) + randn(10, ) * 0.05

observe!(nystrom, X, y)
observe!(full, X, y)
observe!(sarkka, X, y)

@show nll(nystrom)
@show nll(full)
@show nll(sarkka)

@show @elapsed infer!(MLE2(), nystrom)
@show @elapsed infer!(MLE2(), full)
@show @elapsed infer!(MLE2(), sarkka)

@show nll(nystrom)
@show nll(full)
@show nll(sarkka)

xp = linspace(-4, 4, 100)''
if d == 1
	figure(1);clf();hold(true)
	f = sin(vec(sum(xp,2))) .* exp(-vec(sum(xp, 2)).^2 / 2) .+ cos(vec(sum(xp,2)) * 2) .* exp(-vec(sum(xp, 2)).^2)
	plot(xp, f, "k", linewidth=3)
	# plot(X, y, "bo")

	mus, sigma2s = predict(sarkka, xp)
	plot(xp, mus, "g", linewidth=2)
	mun, sigma2n = predict(nystrom, xp)
	plot(xp, mun, "r", linewidth=2)
	muf, sigma2f = predict(full, xp)
	plot(xp, muf, "b", linewidth=2)

	legend(["function", "sarkka", "nystrom", "full"])
	plot(xp, mun + 2sqrt(sigma2n), "r-")
	plot(xp, mun - 2sqrt(sigma2n), "r-")

	
	plot(xp, muf + 2sqrt(sigma2f), "b--")
	plot(xp, muf - 2sqrt(sigma2f), "b--")

	
	plot(xp, mus + 2sqrt(sigma2s), "g:")
	plot(xp, mus - 2sqrt(sigma2s), "g:")
end


end