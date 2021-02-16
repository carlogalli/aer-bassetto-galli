##
#= Setup Model Structure and Functions =#

# Define directory, load packages
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Distributions, PyPlot, Roots

# Change default PyPlot settings
rc("axes", labelsize=25);   rc("axes", titlesize=25);   rc("legend", fontsize=20);  rc("legend", numpoints=1);  rc("legend", frameon=false)

# Create a structure with all model parameters
struct Model
    θbar::Real  # Default cutoff
    μ0::Real
    α0::Real
    β1::Real
    β2::Real
    ψ1::Real
    ψ2::Real
    τ_ρ::Real
    δ::Real
end

# Function that builds a model structure
function Model(; θbar=0., μ0=0., α0=1, β1=1, β2=1, ψ1=1, ψ2=1, τ_ρ=1, δ=0.6)
    return Model(θbar, μ0, α0, β1, β2, ψ1, ψ2, τ_ρ, δ)
end

# Function that flexibly updates model structure by declaring and changing one variable at a time
function new_m(m::Model, var::String, val::Real)
  if var == "θbar"
    return Model(θbar=val, μ0=m.μ0, α0=m.α0, β1=m.β1, β2=m.β2, ψ1=m.ψ1, ψ2=m.ψ2, τ_ρ=m.τ_ρ, δ=m.δ)
  elseif var == "μ0"
    return Model(θbar=m.θbar, μ0=val, α0=m.α0, β1=m.β1, β2=m.β2, ψ1=m.ψ1, ψ2=m.ψ2, τ_ρ=m.τ_ρ, δ=m.δ)
  elseif var == "α0"
    return Model(θbar=m.θbar, μ0=m.μ0, α0=val, β1=m.β1, β2=m.β2, ψ1=m.ψ1, ψ2=m.ψ2, τ_ρ=m.τ_ρ, δ=m.δ)
  elseif var == "β1"
    return Model(θbar=m.θbar, μ0=m.μ0, α0=m.α0, β1=val, β2=m.β2, ψ1=m.ψ1, ψ2=m.ψ2, τ_ρ=m.τ_ρ, δ=m.δ)
  elseif var == "β2"
    return Model(θbar=m.θbar, μ0=m.μ0, α0=m.α0, β1=m.β1, β2=val, ψ1=m.ψ1, ψ2=m.ψ2, τ_ρ=m.τ_ρ, δ=m.δ)
  elseif var == "ψ1"
    return Model(θbar=m.θbar, μ0=m.μ0, α0=m.α0, β1=m.β1, β2=m.β2, ψ1=val, ψ2=m.ψ2, τ_ρ=m.τ_ρ, δ=m.δ)
  elseif var == "ψ2"
    return Model(θbar=m.θbar, μ0=m.μ0, α0=m.α0, β1=m.β1, β2=m.β2, ψ1=m.ψ1, ψ2=val, δ=m.δ)
elseif var == "τ_ρ"
      return Model(θbar=m.θbar, μ0=m.μ0, α0=m.α0, β1=m.β1, β2=m.β2, ψ1=m.ψ1, ψ2=m.ψ2, τ_ρ=val, δ=m.δ)
  end
end

# Second period traders' beliefs variance (σ^2_2 in the paper)
var_2(m::Model) = 1/(m.α0 + m.τ_ρ + m.β2*(1+m.ψ2))

# Variance of new second-period information conditional on first-period information (σ^2_{2|1} in the paper)
var_21(m::Model) = 1/(m.α0 + m.β1*(1+m.ψ1)) + 1/(m.β2*m.ψ2)

# Bayesian weight given to z_1 by first-period traders
w_1(m::Model) = m.β1*(1+m.ψ1) / (m.α0 + m.β1*(1+m.ψ1))

# Bayesian weight given to ρ by second-period traders
w_r(m::Model) = m.τ_ρ / (m.α0 + m.τ_ρ + m.β2*(1+m.ψ2))

# Bayesian weight given to z_2 by second-period traders
w_z2(m::Model) = m.β2*(1+m.ψ2) / (m.α0 + m.τ_ρ + m.β2*(1+m.ψ2))

# Sum of weights given to new information inside the equilibrium price function. This is the numerator of K in equation (22)
W(m::Model) = w_r(m)+w_z2(m)*w_1(m)

# Standard deviation of the distorted probability measure used in the equilibrium price q_1. This is defined in equation (17)
tildesigma1(m::Model) = sqrt(w_z2(m)^2*var_21(m) + w_r(m)^2*(1/m.τ_ρ-1/(m.β1*m.ψ1)) + var_2(m))

# Coefficient K as defined in equation (22)
K(m::Model) = W(m)/tildesigma1(m)

# Function that finds the equilibrium first-period price
function q1(m::Model, z1::Real)
  # conditional price function q_1(z_1)
  return m.δ + (1-m.δ)*cdf(Normal(0,1), (m.μ0 - m.θbar + W(m)*(z1-m.μ0)) / tildesigma1(m))
end

# Endogenous default threshold function 
thetabar(q1::Real) = 1/(q1)

# Function that computes the right-hand side of equation (24), to be used to solve such equation and find q_1
rhs(m::Model, z1::Real, q1::Real) = m.δ + (1-m.δ)*cdf(Normal(0,1), (m.μ0 - thetabar(q1) + W(m)*(z1-m.μ0)) / tildesigma1(m))

# # Function that finds the equilibrium first-period price, in the case of an endogenous default threshold (Section IV)
function q1end(m::Model, z1::Real; err::Float64 = 1., err_tol::Float64 = 1e-16, iter::Int=1, max_iter::Int=100, q0::Float64=m.δ)
    while abs(err) > err_tol
        q1 = rhs(m, z1, q0)
        err = q1-q0
        q0 = q1
        iter += 1
    end
    return q0
end



## 
#= Paper Figure 1: Plot q1 in high/low K cases (more/less sensitive) =#
# Create model instance
m = Model(θbar=0, μ0=0, α0=0.1, β1=1/2, β2=1, ψ1=1/2, ψ2=1/2, τ_ρ=1/4, δ=0) 

figure(1)
clf()
z1g = collect(range(-5, stop=5, length=100))    # Create grid for z_1
plot(z1g, [q1(m, z1) for z1 in z1g], label=L"high $K$", lw=3, "-b")     # Plot q_1 for high β_2 and high K
plot(z1g, [q1(new_m(m, "β2", 0.2*m.β2), z1) for z1 in z1g], label=L"low $K$", lw=3, "--r")  # Plot q_1 for low β_2 and low K

# # Uncomment the following two lines to find and plot the point (̂z) where the two price functions intersect
# zhat = find_zero(x->q1(m, x)-q1(new_m(m, "β2", 0.2*m.β2),x), (-5, 5))
# plot(zhat, q1(m, zhat), "og", ms=10, label=L"q_1(\hat{z};\beta_2)")

# Format graph legends, title and axis labels
fs=20
legend(loc=2, fontsize=fs)
xlim(-4, 4); ylim(0,1)
xlabel(L"z_1", fontsize=fs)
gca()[:xaxis]["set_label_coords"](0.5,-0.01)
gca()[:xaxis]["set_ticks"]([])
ylabel(L"q_1", fontsize=fs)
gca()[:yaxis]["set_label_coords"](-0.01,0.5)
PyPlot.yticks([0,1], (L"\delta",1), fontsize=15)
title(L"q_1(z_1)", fontsize=fs)
gca()["title"]["set_y"](1.02)

# Save figure
savefig("fig1.pdf", orientation="landscape")




## 
#= Paper Figure 2: plot K, the coefficient (of z1, inside q1 CDF's arg) as a function of β_2 =#

# Define common parameters to use in both panels
β1 = 1
ψ = 3
fs = 20     # fontsize

# Grid for β_2
β2s = collect(1e-10:0.01:β1*2)

figure(2)
clf()

suptitle(L"$t=1$ Price Responsiveness", fontsize=fs, y=0.97)

# Left panel
subplot(1,2,1)
m = Model(α0=5, β1=β1, β2=β1, ψ1=ψ, ψ2=ψ, τ_ρ=β1*ψ)
plot(β2s, [K(new_m(m, "β2", x)) for x in β2s], "-b", lw=3, label=L"K")

# Format left panel
xlim(0,1.3)
xlabel(L"\beta_2/\beta_1", fontsize=fs)
gca()[:xaxis]["set_label_coords"](0.465,-0.04); PyPlot.xticks([0,1])
gca()[:tick_params]("both", direction="out", length=4, width=1, colors="k", labelsize=9)
gca()[:yaxis]["set_ticks"]([])
plot((β1, β1), (0,3), "--g", lw=3)
legend(loc=2, fontsize=fs)
ylim(1.05, 1.12)
xlim(0,1.3)

# Right panel
subplot(1,2,2)
m = Model(α0=1, β1=β1, β2=β1, ψ1=ψ, ψ2=ψ, τ_ρ=β1*ψ)
β2hat = (m.ψ1*m.β1*(1+m.ψ1) - m.α0*(2*m.ψ2-m.ψ1))/((1+m.ψ2)*(1+m.ψ1+2*m.ψ2))
β2hat>0 ? str="\\hat \\beta_2>0" : str="\\hat \\beta_2<0"
l, = plot(β2s, [K(new_m(m, "β2", x)) for x in β2s], "-b", lw=3, label=L"K")
c = l[:get_color]()          # get color

# Format right panel
xlabel(L"\beta_2/\beta_1", fontsize=fs)
gca()[:xaxis]["set_label_coords"](0.465,-0.04); PyPlot.xticks([0,1])
gca()[:tick_params]("both", direction="out", length=4, width=1, colors="k", labelsize=9)
gca()[:yaxis]["set_ticks"]([])
plot((β1, β1), (0,3), "--g", lw=3)
ylim(1.48, 1.56)
xlim(0,1.3)
legend(loc=2, fontsize=fs)

# Save figure
savefig("fig2.pdf", orientation="landscape")

## 
#= Paper Figure 3: plot endogenous price function with multiple crossings changing β_2 =#

# Grid for z_1
z1g = collect(range(-20, stop=30, length=10000))

# Model instance
m = Model(μ0=1.22, α0=15, β1=10, β2=10, ψ1=0.1, ψ2=0.1, τ_ρ=10*0.1, δ=0.654)

figure(3)
clf()
title(L"q_1(z_1)", fontsize=fs)
plot(z1g, [q1end(m, z1) for z1 in z1g], label=L"high $\beta_2$", lw=3, "-b")
plot(z1g, [q1end(new_m(m, "β2", 1e-3), z1) for z1 in z1g], label=L"low $\beta_2$", lw=3, "--r")

# Format graph
fs=20 
xlim(-5,8)
xlabel(L"z_1", fontsize=fs);  gca()[:xaxis]["set_label_coords"](0.5,-0.01); gca()[:xaxis]["set_ticks"]([])
ylabel(L"q_1", fontsize=fs); gca()[:yaxis]["set_label_coords"](-0.01,0.5)
PyPlot.yticks([m.δ,1], (L"\delta",1), fontsize=15)
legend()

# Save figure
savefig("fig3.pdf", orientation="landscape")