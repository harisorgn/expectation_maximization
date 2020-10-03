
function run_EM!(df, model_llhood_func, prior_param_v, subj_param_lb_v, subj_param_ub_v)
	#-------------------------------------------------------------------------------------------------------------------
	# df : data frame, must contain "ID" and "reward" columns for subject IDs and trial responses respectively
	# model_llhood_func : loglikelihood function of responses given subject parameters, the decision making model
	# prior_param_v : initial guess for θ, the parameters of the prior Gaussian distribution. 
	#				  Array of consecutive [μ, σ2] pairs, as many as the subject parameters (decision model parameters)
	# subj_param_lb_v : array of lower bounds for subject parameters
	# subj_param_ub_v : array of upper bounds for subject parameters
	#-------------------------------------------------------------------------------------------------------------------
	# Returns prior_param_v : array of optimal prior parameters, in-place of the prior_param_v input, same format of [μ, σ2] pairs

	ID_v = unique(df[!, :ID])
	n_subj = length(ID_v)

	# Stopping criteria 
	max_iter = Int(1e5)
	abs_tol = 1e-8
	rel_tol = 1e-8

	# Initial guess for each participant's parameters to be used for the Laplace approximation optimisation :
	subj_param_m = repeat(prior_param_v[1:2:end]', outer = [n_subj, 1])

	i = 1 
	converged_flag = false

	while !converged_flag && (i <= max_iter)

		prev_prior_param_v = prior_param_v[:]

		subj_param_m = repeat(prior_param_v[1:2:end]', outer = [n_subj, 1])

		prior_param_v .= 0.0

		s = 1

		for subj_ID in ID_v
			
			data_v = df[df[!,:ID] .== subj_ID, :reward]
			
			func = TwiceDifferentiable(x -> neg_log_map(model_llhood_func, x, data_v, prev_prior_param_v), 
										subj_param_m[s, :] ; autodiff=:forward);

			func_c = TwiceDifferentiableConstraints(subj_param_lb_v, subj_param_ub_v)

			res = optimize(func, func_c, subj_param_m[s, :], IPNewton())

			subj_param_v = Optim.minimizer(res)
			var_cov_m = inv(hessian!(func, subj_param_v))
			
			prior_param_v[1:2:end] .+= subj_param_v

			prior_param_v[2:2:end] .+= subj_param_v.^2.0 + diag(var_cov_m)

			s += 1
		end

		prior_param_v[1:2:end] /= length(ID_v)
		prior_param_v[2:2:end] /= length(ID_v)
		prior_param_v[2:2:end] -= prior_param_v[1:2:end].^2.0

		if (sum(abs.(prior_param_v - prev_prior_param_v) .<= abs_tol) == length(prior_param_v) ||
			sum(abs.((prior_param_v - prev_prior_param_v)./prior_param_v) .<= rel_tol) == length(prior_param_v))

			converged_flag = true

		end
		
		i += 1
	end

	println("converged : ", converged_flag)
	println("Max likelihood prior parameters : ")
	for i = 1:2:length(prior_param_v)
		println(prior_param_v[i], "  ", sqrt(prior_param_v[i+1]))
	end

	return prior_param_v
end
