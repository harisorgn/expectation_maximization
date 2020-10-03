using Optim, NLSolversBase, LineSearches
using LinearAlgebra
using CSV
using DataFrames
using Random
using Distributions

include("models.jl")
include("opt_em.jl")

function main()
	
	#=
	# Read actual ABT data
	df = CSV.read("jp4.csv", copycols = true, delim = ", ")

	append!(df, CSV.read("jh3.csv", copycols = true, delim = ", "))

	append!(df, CSV.read("jb12.csv", copycols = true, delim = ", "))
	=#

	# Generate artificial ABT data to test goodness of fit
	μ_beta = 3.5
	μ_n = 0.5
	μ_lapse = 0.3

	σ_beta = 1.0
	σ_n = 0.2
	σ_lapse = 0.1

	#--------------------------------------------------------
	# Generate & fit softmax model
	df = generate_softmax_df(μ_beta, σ_beta, μ_n, σ_n, 20, 50)

	run_EM!(df, loglikelihood_softmax, [μ_beta, σ_beta^2.0, μ_n, σ_n^2.0], 
									   [0.0, 0.0], [Inf, 1.0])
	#--------------------------------------------------------

	#=
	#--------------------------------------------------------
	# Generate & fit lapse model
	df = generate_lapse_df(μ_lapse, σ_lapse, μ_n, σ_n, 50, 50)

	run_EM!(df, loglikelihood_lapse, [μ_lapse, σ_lapse^2.0, μ_n, σ_n^2.0], 
									 [0.0, 0.0], [1.0, 1.0])
	#--------------------------------------------------------
	=#

	#=
	#--------------------------------------------------------
	# Generate & fit softmax+lapse model
	df = generate_softmax_lapse_df(μ_beta, σ_beta, μ_lapse, σ_lapse, μ_n, σ_n, 50, 50)

	run_EM!(df, loglikelihood_softmax_lapse, [μ_beta, σ_beta^2.0, μ_lapse, σ_lapse^2.0, μ_n, σ_n^2.0], 
											 [0.0, 0.0, 0.0], [Inf, 1.0, 1.0])
	#--------------------------------------------------------
	=#
end

main()