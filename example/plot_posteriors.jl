#reload("../../GaussianProcess/src/GaussianProcess.jl")
module t
using PyPlot
import GaussianProcess
#rf = SKLearnRF.RandomForest(100)
# fitc = GPyFITC.FITCModel(50)

k = GaussianProcess.MaternIso32(0, 0)
lik = GaussianProcess.GaussianLikelihood(0)
fullgp = GaussianProcess.FullModel(k, lik)
c = GaussianProcess.OneD(13)

X = linspace(-10, 10, 7)''
X = vcat(X, linspace(0, 3, 10)'')
y = exp(-(X[:].^2) / 5)  - exp(-((X[:] - 4).^2) / 3)
Xp = linspace(-10, 10, 500)''


close(1);figure(1, figsize=(6, 9.6));clf();
function plot_it(X, y, Xp, mu, sigma2)
	plot(X, y, "kx")
	plot(Xp, yp)
	fill_between(Xp[:], (yp + 2sqrt(sigma2))[:], (yp - 2sqrt(sigma2))[:], alpha=0.5)
	locs, labels = yticks()
	yticks([locs[1]*1.5, 0, locs[end]*1.5])
	xticks([])
	ylim([-2.5, 2.5])
	xlim([-10, 10])
end

inducing = 50
subplot(4, 1, 1)
title("Full GP")
GaussianProcess.observe!(fullgp, X, y)
GaussianProcess.infer!(GaussianProcess.MLE2([-10, -10, -10], 
			[10, 10, 10]), fullgp)
yp, sigma2 = GaussianProcess.predict(fullgp, Xp)
plot_it(X, y, Xp, yp, sigma2)

subplot(4, 1, 2)
title("Nyström ($inducing basis functions)")
nystrom = GaussianProcess.NystromModel(k, lik, [10.], [-10.], neig_samples=inducing)
GaussianProcess.observe!(nystrom, X, y)
GaussianProcess.infer!(GaussianProcess.MLE2([-10, -10, -10], 
			[10, 10, 10]), nystrom)
yp, nsigma2 = GaussianProcess.predict(nystrom, Xp)
plot_it(X, y, Xp, yp, nsigma2)

subplot(4, 1, 3)
title("Laplacian ($inducing basis functions)")
laplacian = GaussianProcess.LaplacianModel(k, lik, c, inducing)
GaussianProcess.observe!(laplacian, X, y)
GaussianProcess.infer!(GaussianProcess.MLE2([-10, -10, -10], 
			[10, 10, 10]), laplacian)
yp, lsigma2 = GaussianProcess.predict(laplacian, Xp)
plot_it(X, y, Xp, yp, lsigma2)

subplot(4, 1, 4)
u = linspace(-10, 10, inducing)''
title("FITC ($inducing inducing points)")
fitc = GaussianProcess.FITCModel(k, lik, u)
GaussianProcess.observe!(fitc, X, y)
GaussianProcess.infer!(GaussianProcess.MLE2(
			[-10, -10, -10, -10*ones(size(u, 1))], 
			[10, 10, 10, 10*ones(size(u, 1))]), fitc)
yp, sigma2 = GaussianProcess.predict(fitc, Xp)
plot_it(X, y, Xp, yp, sigma2)

tight_layout()

end
