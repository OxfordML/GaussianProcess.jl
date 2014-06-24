reload("../src/GaussianProcess.jl")
module t
using GaussianProcess
d = 2
k = SqExpIso(log(0.8), log(1))
lik = GaussianLikelihood(log(sqrt(0.5)))
m1 = NystromModel(k, lik)
m2 = FullModel(k, lik)
m3 = LaplacianModel(k, lik, Cartesian(d, 4), 16)

X = (rand(10, d) -0.5) * 6
y = sin(vec(sum(X,2)))

observe!(m1, X, y)
observe!(m2, X, y)
observe!(m3, X, y)

@show nll(m1)
@show nll(m2)
@show nll(m3)


end