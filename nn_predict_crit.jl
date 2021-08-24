using Flux, DelimitedFiles, CUDA, Surrogates, PyPlot, BSON
xpu = cpu
plt.style.use("seaborn-paper")
loaddir = "/home/chris/Documents/conductionblock/data/bikt/set/1/theta/"

crit_dns = []
for th in 0:18
	tmp = readdlm("$(loaddir)$(th)/crit_dns")
	push!(crit_dns, tmp[2:end,1:end-1])
end
crit_dns = vcat(crit_dns...)

testyIndices = Int.(round.(range(1, size(crit_dns,1), length=10)))
trainIndices = []
for m in 1:size(crit_dns,1)
	if m ∉ testyIndices
		push!(trainIndices,m)
	end
end
print("(training, testing, total) = ($(length(trainIndices)), $(length(testyIndices)), $(size(crit_dns,1)))\n")
x = Float32.(collect(transpose(crit_dns[trainIndices,[1,3]]))) 	|> xpu
y = Float32.(collect(transpose(crit_dns[trainIndices,2]))) 		|> xpu
X = Float32.(collect(transpose(crit_dns[testyIndices,[1,3]]))) 	|> xpu
Y = Float32.(collect(transpose(crit_dns[testyIndices,2]))) 		|> xpu

model = Chain(
	Dense( 2, 16, relu),
	Dense(16, 16, relu),
	Dense(16, 16, relu),
	Dense(16, 16, relu),
	Dense(16,  1),
	) |> xpu

opt = ADADelta() #ADAM(0.001, (0.9, 0.999))
loss(x, y) = Flux.Losses.mae(model(x), y)
ps = Flux.params(model)
dl = Flux.DataLoader((x, y), batchsize=64, shuffle=true) |> xpu
Flux.@epochs 20000 Flux.train!(loss, ps, dl, opt; cb = Flux.throttle(() -> @show((loss(x,y), loss(X, Y))), 10))

plt.figure(figsize=(4,2))
plt.plot(x[1,1:15], y[1:15], "ok", label="DNS")
xx = 10.0.^(range(log10(minimum(x[1,1:15])), log10(maximum(x[1,1:15])), length=129))
yy = collect(transpose(model(collect(transpose(hcat(xx, zeros(length(xx))))))))
plt.plot(xx, yy, "--C0", label="NN($(length(vcat([ps[n][:] for n=1:length(ps)]...))))")

model |> cpu; x |> cpu; y |> cpu; dl |> cpu;
BSON.@save "nn_Us.bson" model

model2 = RadialBasis([(x[1,n],x[2,n]) for n in 1:size(x,2)], [(y[n]) for n in 1:size(x,2)], [minimum(x[1,:]),minimum(x[2,:])], [maximum(x[1,:]),maximum(x[2,:])])

yy = model2.([(xx[n],0.0) for n in 1:length(xx)])
plt.plot(xx, yy, "--C1", label="Surrogate")
plt.xscale("log"); plt.yscale("symlog")
plt.legend(loc=0, edgecolor="none")
plt.title("\$ \\theta = 0 \$")
plt.xlabel("\$ x_s \$")
plt.ylabel("\$ U_s(x_s,\\theta) \$")
plt.savefig("./comparison.svg", bbox_inches="tight")
plt.close()

# use contour plots to show approximate functions in (xs,th) plane
fig, axs = plt.subplots(1,3, figsize=(9,3), sharex=true, sharey=true, constrained_layout=true)
axs[1].plot(x[1,:], x[2,:], ".w", markersize=2)
axs[1].tricontour( x[1,:], x[2,:], y[:], levels=14, vmin=minimum(y[:]), vmax=maximum(y[:]), linewidths=0.5, colors="k")
img = axs[1].tricontourf(x[1,:], x[2,:], y[:], levels=14, vmin=minimum(y[:]), vmax=maximum(y[:]), cmap="Oranges")
axs[1].set_title("Triangulation \$ U(x_s,\\theta) \$")
axs[1].set_xlabel("\$ x_s \$")
axs[1].set_ylabel("\$ \\theta \$")
axs[1].set_xscale("log")
Y = model(x)
axs[2].plot(x[1,:], x[2,:], ".w", markersize=2)
axs[2].tricontour( x[1,:], x[2,:], Y[:], levels=14, vmin=minimum(y[:]), vmax=maximum(y[:]), linewidths=0.5, colors="k")
axs[2].tricontourf(x[1,:], x[2,:], Y[:], levels=14, vmin=minimum(y[:]), vmax=maximum(y[:]), cmap="Oranges")
axs[2].set_title("Neural Network \$ U(x_s,\\theta) \$")
axs[2].set_xlabel("\$ x_s \$")
#axs[2].set_ylabel("\$ \\theta \$")
#axs[2].set_xscale("log")
Y = model2.([(x[1,n],x[2,n]) for n in 1:size(x,2)])
axs[3].plot(x[1,:], x[2,:], ".w", markersize=2)
axs[3].tricontour( x[1,:], x[2,:], Y[:], levels=14, vmin=minimum(y[:]), vmax=maximum(y[:]), linewidths=0.5, colors="k")
axs[3].tricontourf(x[1,:], x[2,:], Y[:], levels=14, vmin=minimum(y[:]), vmax=maximum(y[:]), cmap="Oranges")
axs[3].set_title("Surrogate Model \$ U(x_s,\\theta) \$")
axs[3].set_xlabel("\$ x_s \$")
#axs[2].set_ylabel("\$ \\theta \$")
#axs[2].set_xscale("log")
plt.colorbar(img, ax=axs, aspect=40, label="\$ U(x_s,\\theta) \$") 
plt.savefig("./contour_comparison.svg", bbox_inches="tight")
plt.close()

