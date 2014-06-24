# function cf(k::MaternIso32, s, d) 
# 	l = exp(k.log_l)
# 	nu = 3//2
# 	((2*nu)/l^2 + s^2)^(-1/2 - nu)/(l^(2*nu)*gamma(nu))::Float64
# end
# function S(k::MaternIso32, ω)
# 	s = ω * 2pi
# 	l = exp(k.log_l)
# 	scale = exp(2k.log_s)
# 	nu = 3//2
# 	scale*(2^(1 + (nu))*sqrt(pi)*gamma(1/2 + (nu))*((2*(nu))/(l)^2 + (s)^2)^(-1/2 - (nu)))/(gamma(nu)*(l)^(2*(nu)))::Float64
# end

function S(k::MaternIso32, ω)
	s = ω * 2pi
	l = exp(k.log_l) / (2pi)
	scale = exp(2k.log_s) * 4pi
	nu = 3//2
	scale*(2^(1 + (nu))*sqrt(pi)*gamma(1/2 + (nu))*((2*(nu))/(l)^2 + (s)^2)^(-1/2 - (nu)))/(gamma(nu)*(l)^(2*(nu)))::Float64
end
# function dSdl(k::MaternIso32, ω)
# 	s = ω * 2pi 
# 	l = exp(k.log_l)
# 	nu = 3//2
# 	l * (cf(k, s) * (-((2^(2 + nu)*sqrt(pi)*nu^(1 + nu)*(-1 + l^2*s^2)*gamma(1/2 + nu))/(2*l*nu + l^3*s^2))))::Float64
# end

# function dSdnu(k::MaternIso32, ω)
# 	s =  ω * 2pi
# 	l = exp(k.log_l)
# 	nu = 3//2
# 	nu * (cf(k, s) * (2^(1 + nu)*sqrt(pi)*nu^nu*gamma(1/2 + nu)*(-1 + l^2*s^2 + nu*log(4) - 2*(2*nu + l^2*s^2)*log(l) + 2*nu*log(nu) + l^2*s^2*log(2*nu) - 
#         (2*nu + l^2*s^2)*log((2*nu)/l^2 + s^2) - (2*nu + l^2*s^2)*(digamma(nu) - digamma(1/2 + nu))))/(2*nu + l^2*s^2))::Float64
# end

# function S(k::SqExpIso, s)
# 	# s = 2pi * omega/
# 	l = exp(2k.log_l)
# 	scale = exp(2k.log_s)
# 	scale*(2pi*l^2)^2 * exp(- 2pi^2 * l^2 * s^2)
# end
