# linear_functions_2d.py


import numpy as np
from scipy.stats import norm


# ----------------------------------------
# Feature-related functions


# Gaussian features with specified centers and equal widths
def phi_gauss(t, t_proto, dt_proto, sig_scale=1.):
    t_ = t[:,None]; mu_ = t_proto[None,:]; dmu_ = dt_proto[None,:]
    phi = norm.pdf(t_, loc=mu_, scale=sig_scale*dmu_)   # shape: num_t, F
    return phi


# Fourier features    **NOT RANDOM**
# sines and cosines, starting from lowest order
def phi_fourier(t, F, t_min, t_max):
    L = t_max - t_min
    
    num_sin = (F-1)//2
    num_cos = F - 1 - num_sin

    n_sin = np.arange(1, num_sin + 1)
    n_cos = np.arange(1, num_cos + 1)
    
    phi_const = np.ones((t.shape[0], 1))
    phi_sin = np.sin( 2*np.pi*n_sin[None,:]*t[:,None]/L  )
    phi_cos = np.cos( 2*np.pi*n_cos[None,:]*t[:,None]/L  )
    
    phi = np.concatenate( (phi_const, phi_cos, phi_sin), axis=-1)
    return phi
        

# get overall feature set by combining x, y, and t features
def get_phi_crossed_func_2d(z, t, phi_funcs, F):
    x, y = z.T
    phi_x_func, phi_y_func, phi_t_func = phi_funcs
    
    phi = (phi_x_func(x)[:,:,None,None]*phi_y_func(y)[:,None,:,None]*phi_t_func(t)[:,None,None,:])
    phi = phi.reshape(phi.shape[0], F)
    return phi


# get grids of pre-evaluated features
def get_phi_crossed_grid(x, y, t, phi_funcs, F):
    phi_x_func, phi_y_func, phi_t_func = phi_funcs
    
    phi_x = phi_x_func(x); num_x = phi_x.shape[0]
    phi_y = phi_y_func(y); num_y = phi_y.shape[0]
    phi_t = phi_t_func(t); num_t = phi_t.shape[0]
    
    phi = phi_x[:,None,None,:,None,None]*phi_y[None,:,None,None,:,None]*phi_t[None,None,:,None,None,:]
    phi = phi.reshape((num_x, num_y, num_t, F))
    return phi



# ----------------------------------------
# Training and sampling


# 'train' a linear model denoiser via analytic result, which requires computing matrices, inverting a covariance, and multiplying matrices
def train_phi_denoise(x_data, phi_func, num_train_samples, num_models, time_funcs, eps_reg=0.1):
    alpha_func, sig_sq_func = time_funcs['alpha_func'], time_funcs['sig_sq_func']
    eps, T = time_funcs['eps'], time_funcs['T']
    
    M, D = x_data.shape
        
    m_samp = np.random.randint(low=0, high=M, size=(num_models, num_train_samples))   # pick mixture components
    x0_samp = x_data[m_samp]
    t_samp = np.random.uniform(eps, T, size=(num_models, num_train_samples))
    x_samp = alpha_func(t_samp)[...,None]*x0_samp + np.sqrt(sig_sq_func(t_samp))[...,None]*np.random.normal(size=(num_models, num_train_samples, D))
    
    x_samp_ = x_samp.reshape((num_models*num_train_samples,D)); t_samp_ = t_samp.reshape(num_models*num_train_samples)
    phi_samp = phi_func(x_samp_, t_samp_)
    phi_samp = phi_samp.reshape(num_models, num_train_samples, phi_samp.shape[-1])

    lamb_tilde = ((alpha_func(t_samp)**2)/sig_sq_func(t_samp))[...,None]
    Sig_phi = ((phi_samp * lamb_tilde).transpose(0, 2, 1) @ phi_samp)/num_train_samples
    A_phi = ((x0_samp * lamb_tilde).transpose(0, 2, 1) @ phi_samp)/num_train_samples
    print('computed empirical matrices')
    Sig_phi_inv = np.linalg.inv(Sig_phi + eps_reg*np.eye(Sig_phi.shape[-1])[None,:,:])
    print('inverted Sig_phi')
    W = A_phi @ Sig_phi_inv
    print('computed W')
    return W, Sig_phi, A_phi



# simple implementation of sampling via first-order PF-ODE integration
# grid of pre-evaluated denoiser values used to avoid many function calls
def sample_PF_ODE_2d_batched(num_samples, x, y, t, time_funcs, mu_est):
    num_x = len(x); num_y = len(y); num_models = mu_est.shape[0]
    b, g_sq, a, sig_sq = time_funcs['beta_t'], time_funcs['g_sq_t'], time_funcs['alpha_t'], time_funcs['sig_sq_t']

    x_t = np.random.normal(loc=0, scale=np.sqrt(sig_sq[0]), size=(num_models, num_samples))  # Init state
    y_t = np.random.normal(loc=0, scale=np.sqrt(sig_sq[0]), size=(num_models, num_samples)) 
    dt = np.roll(t,-1) - t; num_tsteps = len(t)-1         # get time steps
    b_ = np.arange(num_models)[:, None]
    
    # Euler steps
    for k in range(num_tsteps):
        
        # Find index corresponding to current state
        i_ = np.clip(np.searchsorted(x, x_t), None, num_x-1) #; inds[k] = i_
        j_ = np.clip(np.searchsorted(y, y_t), None, num_y-1)
        
        x_t = x_t - ( b[k]*x_t + (g_sq[k]/2)*((a[k]*mu_est[b_, 0, i_, j_, k] - x_t )/sig_sq[k]))*dt[k] 
        y_t = y_t - ( b[k]*y_t + (g_sq[k]/2)*((a[k]*mu_est[b_, 1, i_, j_, k] - y_t )/sig_sq[k]))*dt[k] 

    return x_t, y_t

