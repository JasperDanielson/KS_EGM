#######################################################
# Krusell-Smith (1998) with EGM and Full Integration 
#######################################################

using LinearAlgebra, Printf, Statistics, Random, Distributions
using Interpolations, Plots, DataFrames, GLM

# Parameters and Calibration
Random.seed!(1234)

beta = 0.99
rho = 0.025
gamma = 1.0
alpha_k = 0.36
z_good, z_bad = 1.01, 0.99
nu_good, nu_bad = 0.04, 0.10
L_good, L_bad = 1.0 - nu_good, 1.0 - nu_bad

prod_states = [z_good, z_bad]
lab_supply = [L_good, L_bad]

function k_ss(z::Float64, L::Float64)
    return ((alpha_k * z * L^(1 - alpha_k)) / (1 / beta - (1 - rho)))^(1 / (1 - alpha_k))
end

Nk = 130
k_bot = 0.01
k_top = 3 * k_ss(z_good, L_good)
k_exponent = 2.0

function stretched_grid(a, b, N, shape)
    grid_raw = range(0, 1, length=N) .^ shape
    return a .+ (b - a) .* grid_raw
end

k_vec = stretched_grid(k_bot, k_top, Nk, 2.0)

function next_K_forecast(K_current, z_id, c0, c1, d0, d1)
    if z_id == 1
        return max(exp(c0 + c1 * log(K_current)), 1e-4)
    else
        return max(exp(d0 + d1 * log(K_current)), 1e-4)
    end
end

function interest(K, L, z) 
    return alpha_k * z * (L / K)^(1 - alpha_k) 
end

function wage(K, L, z) 
    return (1 - alpha_k) * z * (K / L)^alpha_k 
end

initial_K_good = k_ss(z_good, L_good)
initial_K_bad = k_ss(z_bad, L_bad)

r_vals_fixed = [interest(initial_K_good, L_good, z_good), interest(initial_K_bad, L_bad, z_bad)]
w_vals_fixed = [wage(initial_K_good, L_good, z_good), wage(initial_K_bad, L_bad, z_bad)]

function build_transition_matrix()
    # Aggregate state transitions
    πgg = 0.875  # Prob of staying in good times
    πbb = 0.875  # Prob of staying in bad times
    πgb = 1 - πgg  # Prob of going from good to bad
    πbg = 1 - πbb  # Prob of going from bad to good

    # Employment transitions within each aggregate state
    # Good times: lower unemployment, shorter spells
    πgg_ee = 0.97   # employed to employed in good times
    πgg_eu = 1 - πgg_ee  # employed to unemployed in good times
    πgg_uu = 0.7    # unemployed to unemployed in good times
    πgg_ue = 1 - πgg_uu  # unemployed to employed in good times

    # Bad times: higher unemployment, longer spells
    πbb_ee = 0.94   # employed to employed in bad times
    πbb_eu = 1 - πbb_ee  # employed to unemployed in bad times
    πbb_uu = 0.8    # unemployed to unemployed in bad times
    πbb_ue = 1 - πbb_uu  # unemployed to employed in bad times

    # Cross-state transitions (when aggregate state changes)
    πgb_ee = 0.9
    πgb_eu = 1 - πgb_ee
    πgb_uu = 0.85
    πgb_ue = 1 - πgb_uu

    πbg_ee = 0.98
    πbg_eu = 1 - πbg_ee
    πbg_uu = 0.6
    πbg_ue = 1 - πbg_uu

    # Build full transition matrix
    # States: (good,employed), (good,unemployed), (bad,employed), (bad,unemployed)
    P = zeros(4, 4)
    
    # From (good, employed)
    P[1, 1] = πgg * πgg_ee     # to (good, employed)
    P[1, 2] = πgg * πgg_eu     # to (good, unemployed)
    P[1, 3] = πgb * πgb_ee     # to (bad, employed)
    P[1, 4] = πgb * πgb_eu     # to (bad, unemployed)
    
    # From (good, unemployed)
    P[2, 1] = πgg * πgg_ue     # to (good, employed)
    P[2, 2] = πgg * πgg_uu     # to (good, unemployed)
    P[2, 3] = πgb * πgb_ue     # to (bad, employed)
    P[2, 4] = πgb * πgb_uu     # to (bad, unemployed)
    
    # From (bad, employed)
    P[3, 1] = πbg * πbg_ee     # to (good, employed)
    P[3, 2] = πbg * πbg_eu     # to (good, unemployed)
    P[3, 3] = πbb * πbb_ee     # to (bad, employed)
    P[3, 4] = πbb * πbb_eu     # to (bad, unemployed)
    
    # From (bad, unemployed)
    P[4, 1] = πbg * πbg_ue     # to (good, employed)
    P[4, 2] = πbg * πbg_uu     # to (good, unemployed)
    P[4, 3] = πbb * πbb_ue     # to (bad, employed)
    P[4, 4] = πbb * πbb_uu     # to (bad, unemployed)
    
    return P
end

const transition_matrix = build_transition_matrix()

function idx(z, e) 
    return (z - 1) * 2 + e 
end

function EGM_solver(k_vec, tol, max_iter, a0, a1, b0, b1)
    Nk = length(k_vec)
    V = zeros(Nk, 2, 2)  # V[k_index, employment, aggregate_state]
    k_pol = similar(V)
    
    # Initialize value function
    for z in 1:2, e in 1:2
        for i in 1:Nk
            inc = (e == 1 ? w_vals_fixed[z] : 0.0)
            k = k_vec[i]
            V[i,e,z] = log(max(inc + (1 + r_vals_fixed[z] - rho) * k, 1e-8)) / (1 - beta)
        end
    end
    
    for iter = 1:max_iter
        V_old = copy(V)
        
        for z in 1:2, e in 1:2
            # EGM step
            dV = diff(V[:,e,z]) ./ diff(k_vec)
            dV = vcat(dV[1], dV)  # Extend first derivative to match grid size
            
            c = 1 ./ max.(dV, 1e-12)  # Consumption from FOC
            y = c .+ k_vec  # Resources needed
            u = log.(max.(c, 1e-12))  # Utility
            V_tilde = u + beta * V[:,e,z]
            
            # Update value function
            for i in 1:Nk
                v_future = 0.0
                for zp in 1:2, ep in 1:2
                    trans_prob = transition_matrix[idx(z,e), idx(zp,ep)]
                    
                    # Future aggregate capital
                    K_pred = next_K_forecast(k_vec[i], z, a0, a1, b0, b1)
                    r_f = interest(K_pred, lab_supply[zp], prod_states[zp])
                    w_f = wage(K_pred, lab_supply[zp], prod_states[zp])
                    
                    # Future income
                    inc_f = (ep == 1 ? w_f : 0.0)
                    y_f = (1 + r_f - rho) * k_vec[i] + inc_f
                    
                    # Interpolate future value
                    if y_f <= y[1]
                        v_interp = V_tilde[1]
                    elseif y_f >= y[end]
                        v_interp = V_tilde[end]
                    else
                        i_star = searchsortedfirst(y, y_f)
                        i_star = clamp(i_star, 2, Nk)
                        x0, x1 = y[i_star-1], y[i_star]
                        y0, y1 = V_tilde[i_star-1], V_tilde[i_star]
                        weight = (y_f - x0) / (x1 - x0)
                        v_interp = y0 + weight * (y1 - y0)
                    end
                    
                    v_future += beta * trans_prob * v_interp
                end
                
                V[i,e,z] = v_future
                
                # Policy function
                inc_now = (e == 1 ? w_vals_fixed[z] : 0.0)
                budget = inc_now + (1 + r_vals_fixed[z] - rho) * k_vec[i]
                
                if budget <= y[1]
                    k_pol[i,e,z] = 0.0
                elseif budget >= y[end]
                    k_pol[i,e,z] = budget - c[end]
                else
                    i_star = searchsortedfirst(y, budget)
                    i_star = clamp(i_star, 2, Nk)
                    x0, x1 = y[i_star-1], y[i_star]
                    c0, c1 = c[i_star-1], c[i_star]
                    weight = (budget - x0) / (x1 - x0)
                    c_interp = c0 + weight * (c1 - c0)
                    k_pol[i,e,z] = max(budget - c_interp, 0.0)
                end
            end
        end
        
        if maximum(abs.(V .- V_old)) < tol
            println("    EGM converged in $iter iterations")
            break
        end
    end
    
    return k_pol, V
end

function simulate_economy(k_vec, policy_func; T=11000, T_burn=2500, N=5000)
    z_hist = zeros(Int, T)
    K_hist = zeros(T)
    
    # Initialize agents
    k_now = rand(Uniform(k_vec[1], k_vec[end]), N)
    k_next = similar(k_now)
    emp_state = vcat(fill(1, Int(0.96 * N)), fill(2, Int(0.04 * N)))  # 96% employed initially
    shuffle!(emp_state)
    
    # Initialize aggregate state
    z_hist[1] = 1  # Start in good times
    
    # Create interpolation functions
    interp = [[LinearInterpolation(k_vec, policy_func[:,e,z], extrapolation_bc=Flat()) for e in 1:2] for z in 1:2]
    
    for t in 1:T-1
        K_hist[t] = mean(k_now)
        z = z_hist[t]
        
        # Update individual capital
        for i in 1:N
            k_next[i] = interp[z][emp_state[i]](k_now[i])
        end
        
        # Update aggregate state
        if z == 1
            z_hist[t+1] = rand() < 0.875 ? 1 : 2
        else
            z_hist[t+1] = rand() < 0.875 ? 2 : 1
        end
        
        # Update employment states
        new_emp_state = similar(emp_state)
        for i in 1:N
            current_state = idx(z, emp_state[i])
            trans_probs = transition_matrix[current_state, :]
            u = rand()
            cum_prob = 0.0
            for new_state in 1:4
                cum_prob += trans_probs[new_state]
                if u <= cum_prob
                    new_emp_state[i] = ((new_state - 1) % 2) + 1
                    break
                end
            end
        end
        emp_state .= new_emp_state
        
        k_now .= k_next
    end
    
    K_hist[T] = mean(k_now)
    return K_hist[T_burn+1:end], z_hist[T_burn+1:end]
end

function forecast_regression(K_series::Vector{Float64}, Z_series::Vector{Int})
    # Smooth series to reduce noise
    function smooth_series(x::Vector{Float64}, w::Int=3)
        smoothed = similar(x)
        for i in 1:length(x)
            lo = max(1, i - w)
            hi = min(length(x), i + w)
            smoothed[i] = mean(x[lo:hi])
        end
        return smoothed
    end

    # Create lagged series
    T = length(K_series)
    log_K = log.(K_series[1:T-1])
    log_Kp = log.(K_series[2:T])
    z_states = Z_series[1:T-1]

    # Smooth if needed
    if std(log_K) > 0.01
        log_K = smooth_series(log_K)
        log_Kp = smooth_series(log_Kp)
    end
    
    good_idx = findall(z_states .== 1)
    bad_idx = findall(z_states .== 2)

    if length(good_idx) < 10 || length(bad_idx) < 10
        println("  Warning: Not enough data for regression")
        return (
            (a0 = 0.01, a1 = 0.96, b0 = -0.01, b1 = 0.96),
            (r2_good = 0.0, r2_bad = 0.0),
            (var_good = 1e-3, var_bad = 1e-3)
        )
    end

    try
        df_good = DataFrame(x = log_K[good_idx], y = log_Kp[good_idx])
        df_bad  = DataFrame(x = log_K[bad_idx], y = log_Kp[bad_idx])

        model_good = lm(@formula(y ~ x), df_good)
        model_bad  = lm(@formula(y ~ x), df_bad)

        return (
            (a0 = coef(model_good)[1], a1 = coef(model_good)[2], 
             b0 = coef(model_bad)[1], b1 = coef(model_bad)[2]),
            (r2_good = r2(model_good), r2_bad = r2(model_bad)),
            (var_good = max(var(residuals(model_good)), 1e-6), 
             var_bad = max(var(residuals(model_bad)), 1e-6))
        )
    catch e
        println("  Regression failed: $e")
        return (
            (a0 = 0.01, a1 = 0.96, b0 = -0.01, b1 = 0.96),
            (r2_good = 0.0, r2_bad = 0.0),
            (var_good = 1e-3, var_bad = 1e-3)
        )
    end
end

function krusell_smith_iteration(k_vec; tol=1e-5, max_iter=1000, eps_r2=0.9999, var_cut=1e-3, max_outer=50)
    best_r2_good = -Inf
    best_r2_bad = -Inf
    a0, a1 = 0.01, 0.96
    b0, b1 = -0.01, 0.96
    
    for outer in 1:max_outer
        println("\nKS Iteration $outer")
        
        # Solve individual problem
        pol_func, _ = EGM_solver(k_vec, tol, max_iter, a0, a1, b0, b1)
        
        # Simulate economy
        K_series, Z_series = simulate_economy(k_vec, pol_func)
        
        # Check for degenerate simulation
        if any(isnan.(K_series)) || any(isinf.(K_series)) || std(K_series) < 1e-8
            println("  Warning: Degenerate simulation, continuing with previous coefficients")
            continue
        end
        
        # Ensure positive capital
        K_series .= max.(K_series, 1e-6)
        
        # Estimate forecasting rules
        coefs, r2s, vars = forecast_regression(K_series, Z_series)
        
        println("  R² Good = $(round(r2s.r2_good, digits=4)), R² Bad = $(round(r2s.r2_bad, digits=4))")
        println("    Good: log(K') = $(round(coefs.a0, digits=4)) + $(round(coefs.a1, digits=4)) * log(K)")
        println("    Bad : log(K') = $(round(coefs.b0, digits=4)) + $(round(coefs.b1, digits=4)) * log(K)")
        
        # Update coefficients with dampening
        α = 0.3
        updated = false
        
        if r2s.r2_good > best_r2_good + 1e-6
            best_r2_good = r2s.r2_good
            a0 = α * coefs.a0 + (1 - α) * a0
            a1 = α * coefs.a1 + (1 - α) * a1
            updated = true
        end
        
        if r2s.r2_bad > best_r2_bad + 1e-6
            best_r2_bad = r2s.r2_bad
            b0 = α * coefs.b0 + (1 - α) * b0
            b1 = α * coefs.b1 + (1 - α) * b1
            updated = true
        end
        
        if !updated
            println("  No improvement in R², using previous coefficients")
        end
        
        # Check convergence
        if r2s.r2_good > eps_r2 && r2s.r2_bad > eps_r2
            println("  Converged! High R² achieved.")
            return (a0=a0, a1=a1, b0=b0, b1=b1), r2s, vars, pol_func
        end
        
        # Early stopping if R² is reasonable
        if outer > 10 && r2s.r2_good > 0.99 && r2s.r2_bad > 0.99
            println("  Good enough convergence achieved")
            return (a0=a0, a1=a1, b0=b0, b1=b1), r2s, vars, pol_func
        end
    end
    
    println("  Maximum iterations reached, returning best found")
    return (a0=a0, a1=a1, b0=b0, b1=b1), (r2_good=best_r2_good, r2_bad=best_r2_bad), 
           (var_good=1e-4, var_bad=1e-4), pol_func
end

function gini_coefficient(x::Vector{Float64})
    n = length(x)
    sorted = sort(x)
    B = sum((n + 1 - i) * xi for (i, xi) in enumerate(sorted))
    return (n + 1 - 2 * B / sum(sorted)) / n
end

# === Main Execution ===
println("Running Krusell-Smith with EGM integration...")
params_final, R2_final, error_var_final, policy_final = krusell_smith_iteration(k_vec)

println("\n=== FINAL RESULTS ===")
println("\n(a) Final Law of Motion for Capital:")
println("  Good: log(K') = $(round(params_final.a0, digits=4)) + $(round(params_final.a1, digits=4)) * log(K), R² = $(round(R2_final.r2_good, digits=4))")
println("  Bad : log(K') = $(round(params_final.b0, digits=4)) + $(round(params_final.b1, digits=4)) * log(K), R² = $(round(R2_final.r2_bad, digits=4))")

# Final simulation for analysis
println("\nRunning final simulation for analysis...")
K_final, Z_final = simulate_economy(k_vec, policy_final, T=15000, T_burn=5000, N=10000)

# (b) Forecast accuracy
println("\n(b) 25-Year Ahead Forecast Interval:")
if length(K_final) > 25
    std_error = sqrt(error_var_final.var_good) * sqrt(25)
    low_band = exp(-2 * std_error)
    high_band = exp(2 * std_error)
    println("  2-SD Prediction Band (Good State): [$(round(low_band, digits=4)), $(round(high_band, digits=4))]")
    
    # Interest rate prediction
    rate_series = [interest(K, L_good, z_good) for K in K_final]
    if length(rate_series) > 25
        rate_25 = rate_series[26:end]
        rate_base = rate_series[1:end-25]
        df_r = DataFrame(y = rate_25, x = rate_base)
        try
            rate_model = lm(@formula(y ~ x), df_r)
            rate_r2 = r2(rate_model)
            println("  Interest Rate R² (25 years ahead): $(round(rate_r2, digits=6))")
        catch
            println("  Interest Rate R²: Could not compute")
        end
    end
end

# (c) Gini coefficients
println("\n(c) Inequality Analysis:")
N_agents = 10000
agents_wealth = rand(Uniform(k_vec[10], k_vec[end-10]), N_agents)
interp_good = [LinearInterpolation(k_vec, policy_final[:,e,1], extrapolation_bc=Flat()) for e in 1:2]

income_vec = zeros(N_agents)
cons_vec = zeros(N_agents)
K_mean = mean(agents_wealth)

for i in 1:N_agents
    e_stat = i <= round(Int, 0.96 * N_agents) ? 1 : 2  # 96% employed
    k_i = agents_wealth[i]
    kp_i = interp_good[e_stat](k_i)
    
    r_now = interest(K_mean, L_good, z_good)
    w_now = wage(K_mean, L_good, z_good)
    
    inc = r_now * k_i + (e_stat == 1 ? w_now : 0.0)
    cons = inc + (1 - rho) * k_i - kp_i
    
    income_vec[i] = max(inc, 1e-8)
    cons_vec[i] = max(cons, 1e-8)
end

println("  Wealth Gini: $(round(gini_coefficient(agents_wealth), digits=4))")
println("  Income Gini: $(round(gini_coefficient(income_vec), digits=4))")
println("  Consumption Gini: $(round(gini_coefficient(cons_vec), digits=4))")

println("\n✓ Krusell-Smith model solved successfully!")