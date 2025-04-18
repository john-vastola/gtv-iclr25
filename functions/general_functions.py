# general_functions.py


import numpy as np
from scipy.stats import norm


# ----------------------------------------

# specify time, noise schedule
# VP choice used in Figs 2 and 3, EDM not used
def get_time_funcs(eps, T, num_tsteps, params, option='VP'):
    t_lin = np.flip(np.linspace(eps, T, num_tsteps + 1))

    if option=='EDM':
        rho = params['rho']
        A = (eps**(1/rho) - T**(1/rho))/(num_tsteps)
        B = T**(1/rho)
        t = (A*np.arange(num_tsteps+1) + B)**rho
        
        beta_t = np.zeros_like(t)
        g_sq_t = 2*t
        alpha_t = np.ones_like(t)
        sig_sq_t = t**2
        
        alpha_func = lambda t: np.ones_like(t)
        sig_sq_func = lambda t : t**2
        
    elif option=='VP':
        t = t_lin
        beta_min, beta_d = params['beta_min'], params['beta_d']
    
        beta_t = beta_min + beta_d*t
        g_sq_t = 2*beta_t 
        alpha_t = np.exp( - (beta_min*t + (beta_d/2.)*t*t) )
        sig_sq_t = 1 - alpha_t**2
        
        alpha_func = lambda t : np.exp( - (beta_min*t + (beta_d/2.)*t*t) )
        sig_sq_func = lambda t : 1 - alpha_func(t)**2
    lamb_t = sig_sq_t
    
    dt = np.roll(t, +1) - t; dt[0] = dt[1]
    bins_t = np.zeros(num_tsteps+2)
    bins_t[0] = t[0] + 0.5*dt[1]
    bins_t[1:-1] = t[:-1] - 0.5*dt[1:]
    bins_t[-1] = t[-1] - 0.5*dt[-1]
    
    dt = (np.roll(bins_t,+1) - bins_t)[1:]
        
    time_funcs = {'t':t, 't_lin':t_lin, 'beta_t':beta_t, 'g_sq_t':g_sq_t, 'alpha_t':alpha_t, 'sig_sq_t':sig_sq_t, 'lamb_t':lamb_t,
                  'D_t':g_sq_t/2., 'alpha_func':alpha_func, 'sig_sq_func':sig_sq_func, 'eps':eps, 'T':T,
                  'dt':dt, 'bins_t':bins_t}
    return t, t_lin, time_funcs



# get basic info about 1D models, used in Fig. 2 on 1D generalization
def get_basic_distributions_1d(x_data, x, time_funcs):
    alpha_t, sig_sq_t, dt = time_funcs['alpha_t'], time_funcs['sig_sq_t'], time_funcs['dt']
    eps, T = time_funcs['eps'], time_funcs['T']
    std_t = np.sqrt(sig_sq_t)    # std of noise-corrupted data distribution
    dx = x[1] - x[0]
    alpha_t_ = alpha_t[None,:]; std_t_ = std_t[None,:]

    # Compute relevant distributions
    x_ = x[:,None,None]; x_data_ = x_data[None,None,:]
    x_data_norm = norm.pdf(x_, loc=x_data_*alpha_t[None,:,None], scale=std_t[None,:,None])   # shape: (num_x, num_t, num_data)
    prob_t = np.mean(x_data_norm, axis=-1)              # shape: (num_x, num_t)
    p_norm_t = prob_t*dx*dt[None,:]/(T - eps)
    
    eps_stab = 1e-8
    p_bayes = x_data_norm/(np.sum(x_data_norm,axis=-1)[:,:,None] + eps_stab)   # shape: (num_x, num_t, num_data)

    # Compute relevant moments
    mu_bayes = np.sum( x_data_*p_bayes, axis=-1)                                  # shape: (num_x, num_t)
    var_bayes = np.sum( ( (x_data_ - mu_bayes[:,:,None] )**2)*p_bayes, axis=-1)   # shape: (num_x, num_t)

    # Get proxy score variance
    var_proxy = ((alpha_t_/std_t_)**2)*( var_bayes/(std_t_**2) )  # shape: (num_x, num_t)
    
    # Get true score
    score_true = (alpha_t_*mu_bayes - x[:,None])/(std_t_**2)
    
    # Collect results in dictionary
    x_min = np.min(x); x_max = np.max(x)
    dists = {'x':x, 'x_data_norm':x_data_norm, 'prob_t':prob_t, 'p_bayes':p_bayes, 'mu_bayes':mu_bayes, 'var_bayes':var_bayes, 
             'var_proxy':var_proxy, 'score_true':score_true,
             'x_min':x_min, 'x_max':x_max, 'p_norm_t':p_norm_t}
    return dists



# compute D-dim histogram from samples, used in Fig. 3 on 2D generalization
def get_prob_from_samples(data, n_bins, z_min, z_max):
    """
    Compute multiple d-dimensional histograms in a vectorized fashion.

    Parameters:
    - data: ndarray of shape (num_conditions, D, num_samples), where D is the number of dimensions for the histogram.
    - n_bins: int or sequence of ints specifying number of bins per dimension.
    - range_limits: sequence of (min, max) pairs for each dimension.

    Returns:
    - counts: ndarray of shape (num_conditions, n_bins[0], n_bins[1], ..., n_bins[D-1])
    """
    data = np.asarray(data)
    num_conditions, D, num_samples = data.shape

    # Create bins
    if np.isscalar(n_bins):
        n_bins = [n_bins] * D
    n_bins = np.asarray(n_bins)
    bins = [np.linspace(z_min[d], z_max[d], n_bins[d] + 1) for d in range(D)]

    # Compute bin indices for each dimension separately
    idx = [np.searchsorted(bins[d], data[:, d, :], side='right') - 1 for d in range(D)]  # List of arrays (num_conditions, num_samples)
    idx = np.stack(idx, axis=-1)  # Shape: (num_conditions, num_samples, D)

    # Mask out-of-range values
    valid_mask = np.all((idx >= 0) & (idx < n_bins), axis=-1)  # Shape: (num_conditions, num_samples)

    # Compute linearized bin indices per histogram
    strides = np.cumprod([1] + list(n_bins[:-1]))  # Compute strides for each dimension
    flat_idx = np.dot(idx, strides)  # Shape: (num_conditions, num_samples)

    # Apply condition-specific offsets properly
    condition_offsets = np.arange(num_conditions)[:, None] * np.prod(n_bins)  # Shape: (num_conditions, 1)
    flat_idx = flat_idx + condition_offsets  # Broadcasting ensures correct shape

    # Apply valid mask
    flat_idx = flat_idx[valid_mask]  # Shape: (num_valid_entries,)

    # Compute histograms
    counts = np.bincount(flat_idx.ravel(), minlength=num_conditions * np.prod(n_bins))
    counts = counts.reshape((num_conditions,) + tuple(n_bins))

    eps_tol = 1e-15
    counts = counts/(eps_tol + np.sum(counts, axis=tuple(range(1, D)), keepdims=True))
    return counts
    
    