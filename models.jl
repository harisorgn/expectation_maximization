
gauss_pdf(x, m, s) = exp(-((x - m)^2.0)/(2.0*s^2.0)) / (s*sqrt(2.0*pi))

p1_softmax(beta, r1, r2, b1 = 0.0, b2 = 0.0) = exp(beta * (r1 + b1)) / (exp(beta * (r1 + b1)) + exp(beta * (r2 + b2)))

p1_lapse(p_l, r1, r2, b1 = 0.0, b2 = 0.0) = (r1 >= r2) ? (1.0 - p_l + (p_l/2.0)) : p_l/2.0

p1_softmax_lapse(beta, p_l, r1, r2, b1 = 0.0, b2 = 0.0) = (1.0 - p_l) * p1_softmax(beta, r1, r2, b1, b2) + (p_l/2.0)

#------------------
# ABT reward values
const R1 = 1.0
const R2 = 0.0
#------------------

function neg_log_map(model_llhood_func, x, data_v, prior_param_v)

	obj = model_llhood_func(x, data_v)

	for i = 1:2:length(prior_param_v)

		obj += log(gauss_pdf(x[Int(ceil(i/2))], prior_param_v[i], sqrt(prior_param_v[i+1])))

	end
	return -obj
end

loglikelihood_test(x, data_v) = sum(logpdf.(Beta(x[1], x[2]), data_v))

function loglikelihood_softmax(x, choice_v)

	# Log likelihood calculation for softmax model
	# x = [beta, n]

	n_trials = length(choice_v)

	r1_exp = 0.0 
	r2_exp = 0.0

	obj = 0.0

	for i = 1 : n_trials

		p1 = p1_softmax(x[1], r1_exp, r2_exp)

		if choice_v[i] == 1

			r1_exp += x[2] * (R1 - r1_exp)

			obj += log(p1)

		else

			r2_exp += x[2] * (R2 - r2_exp)

			obj += log(1.0 - p1)
		end
	end

	return obj
end

function loglikelihood_lapse(x, choice_v)

	# Log likelihood calculation for lapse model
	# x = [probability of lapse, n]

	n_trials = length(choice_v)

	r1_exp = 0.0 
	r2_exp = 0.0

	obj = 0.0

	for i = 1 : n_trials
		
		p1 = p1_lapse(x[1], r1_exp, r2_exp)

		if choice_v[i] == 1
			
			r1_exp += x[2] * (R1 - r1_exp)

			obj += log(p1)

		else

			r2_exp += x[2] * (R2 - r2_exp)

			obj += log(1.0 - p1)
		end
	end
	return obj
end

function loglikelihood_softmax_lapse(x, choice_v)

	# Log likelihood calculation for softmax + lapse model
	# x = [beta, probability of lapse, n]

	n_trials = length(choice_v)

	r1_exp = 0.0 
	r2_exp = 0.0

	obj = 0.0

	for i = 1 : n_trials

		p1 = p1_softmax_lapse(x[1], x[2], r1_exp, r2_exp)

		if choice_v[i] == 1
			
			r1_exp += x[3] * (R1 - r1_exp)

			obj += log(p1)

		else

			r2_exp += x[3] * (R2 - r2_exp)

			obj += log(1.0 - p1)
		end
	end

	return obj
end

function generate_test_df(μ_a, σ_a, μ_b, σ_b, n_subj, n_trials)

	df = DataFrame(ID = String[], reward = Float64[])

	for s = 1 : n_subj

		a_s = rand(Normal(μ_a, σ_a))
		b_s = rand(Normal(μ_b, σ_b))

		for _ = 1 : n_trials
			push!(df, (string(s), rand(Beta(a_s, b_s))))
		end
	end
	
	return df
end

function generate_softmax_df(μ_beta, σ_beta, μ_n, σ_n, n_subj, n_trials)

	# Generate ABT data using the softmax model with the input μ, σ parameters for #n_subj subjects and #n_trials trials
	# Returns a data frame, same format as true ABT

	df = DataFrame(ID = String[], choice = String[], reward = Int64[])

	max_trials = 20

	rng = RandomDevice()

	for s = 1 : n_subj

		count_trials = 0
		count_correct = 0

		r1_exp = 0.0 
		r2_exp = 0.0

		beta = rand(rng, Normal(μ_beta, σ_beta))
		n = rand(rng, Normal(μ_n, σ_n))

		while (beta < 0.0)
			beta = rand(rng, Normal(μ_beta, σ_beta))
		end

		while (n < 0.0 || n > 1.0)
			n = rand(rng, Normal(μ_n, σ_n))
		end

		for _ = 1 : n_trials

			p1 = p1_softmax(beta, r1_exp, r2_exp)

			if p1 > rand(rng)

				r1_exp += n * (R1 - r1_exp)

				push!(df, (string(s), rand(rng, ["R", "L"]), 1))

				count_correct += 1

			else

				r2_exp += n * (R2 - r2_exp)

				push!(df, (string(s), rand(rng, ["R", "L"]), 0))

				count_correct = 0

			end

			count_trials += 1
		end
	end

	return df
end

function generate_lapse_df(μ_lapse, σ_lapse, μ_n, σ_n, n_subj, n_trials)

	# Generate ABT data using the lapse model with the input μ, σ parameters for #n_subj subjects and #n_trials trials
	# Returns a data frame, same format as true ABT

	df = DataFrame(ID = String[], choice = String[], reward = Int64[])

	max_trials = 20

	rng = RandomDevice()

	for s = 1 : n_subj

		count_trials = 0
		count_correct = 0

		r1_exp = 0.0 
		r2_exp = 0.0

		p_lapse = rand(rng, Normal(μ_lapse, σ_lapse))
		n = rand(rng, Normal(μ_n, σ_n))

		while (p_lapse < 0.0 || p_lapse > 1.0)
			p_lapse = rand(rng, Normal(μ_lapse, σ_lapse))
		end

		while (n < 0.0 || n > 1.0)
			n = rand(rng, Normal(μ_n, σ_n))
		end

		for _ = 1 : n_trials

			p1 = p1_lapse(p_lapse, r1_exp, r2_exp)

			if p1 > rand(rng)

				r1_exp += n * (R1 - r1_exp)

				push!(df, (string(s), rand(rng, ["R", "L"]), 1))

				count_correct += 1

			else

				r2_exp += n * (R2 - r2_exp)

				push!(df, (string(s), rand(rng, ["R", "L"]), 0))

				count_correct = 0

			end

			count_trials += 1
		end
	end

	return df
end

function generate_softmax_lapse_df(μ_beta, σ_beta, μ_lapse, σ_lapse, μ_n, σ_n, n_subj, n_trials)

	# Generate ABT data using the softmax + lapse model with the input μ, σ parameters for #n_subj subjects and #n_trials trials
	# Returns a data frame, same format as true ABT

	df = DataFrame(ID = String[], choice = String[], reward = Int64[])

	max_trials = 20

	rng = RandomDevice()

	for s = 1 : n_subj

		count_trials = 0
		count_correct = 0

		r1_exp = 0.0 
		r2_exp = 0.0

		beta = rand(rng, Normal(μ_beta, σ_beta))
		p_lapse = rand(rng, Normal(μ_lapse, σ_lapse))
		n = rand(rng, Normal(μ_n, σ_n))

		while (beta < 0.0)
			beta = rand(rng, Normal(μ_beta, σ_beta))
		end

		while (p_lapse < 0.0 || p_lapse > 1.0)
			p_lapse = rand(rng, Normal(μ_lapse, σ_lapse))
		end

		while (n < 0.0 || n > 1.0)
			n = rand(rng, Normal(μ_n, σ_n))
		end

		for _ = 1 : n_trials

			p1 = p1_softmax_lapse(beta, p_lapse, r1_exp, r2_exp)

			if p1 > rand(rng)

				r1_exp += n * (R1 - r1_exp)

				push!(df, (string(s), rand(rng, ["R", "L"]), 1))

				count_correct += 1

			else

				r2_exp += n * (R2 - r2_exp)

				push!(df, (string(s), rand(rng, ["R", "L"]), 0))

				count_correct = 0

			end

			count_trials += 1
		end
	end

	return df
end