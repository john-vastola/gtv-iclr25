# linear_functions_1d.py


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
        
        
# get overall feature set by combining x and t features
def get_phi_crossed_func_1d(x, t, phi_funcs, F):
    phi_x_func, phi_t_func = phi_funcs
    
    phi = (phi_x_func(x)[:,:,None]*phi_t_func(t)[:,None,:])
    phi = phi.reshape(phi.shape[0], F)
    return phi


# get grids of pre-evaluated features
def get_phi_crossed_grid_1d(x, t, phi_funcs, F):
    phi_x_func, phi_t_func = phi_funcs
    
    phi_x = phi_x_func(x); num_x = phi_x.shape[0]
    phi_t = phi_t_func(t); num_t = phi_t.shape[0]
    
    phi = phi_x[:,None,:,None]*phi_t[None,:,None,:]
    phi = phi.reshape((num_x, num_t, F))
    return phi



# ----------------------------------------
# Training and sampling


# 'train' a linear model denoiser via analytic result, which requires computing matrices, inverting a covariance, and multiplying matrices
def train_phi_denoise(x_data, phi_func, num_train_samples, num_models, time_funcs, eps_reg=0.1):
    alpha_func, sig_sq_func = time_funcs['alpha_func'], time_funcs['sig_sq_func']
    eps, T = time_funcs['eps'], time_funcs['T']
    
    M = len(x_data)
    m_samp = np.random.randint(low=0, high=M, size=(num_models, num_train_samples))   # pick mixture components
    x0_samp = x_data[m_samp]
    t_samp = np.random.uniform(eps, T, size=(num_models, num_train_samples))
    x_samp = alpha_func(t_samp)*x0_samp + np.sqrt(sig_sq_func(t_samp))*np.random.normal(size=(num_models, num_train_samples))
    
    x_samp_ = x_samp.reshape(num_models*num_train_samples); t_samp_ = t_samp.reshape(num_models*num_train_samples)
    print('start phi samples')
    phi_samp = phi_func(x_samp_, t_samp_)
    phi_samp = phi_samp.reshape(num_models, num_train_samples, phi_samp.shape[-1]) 
    print('got phi samples')
    
    # covariances
    eps_tol = 1e-8; lamb_tilde = (alpha_func(t_samp)**2)/(sig_sq_func(t_samp) + eps_tol)
    Sig_phi = ((phi_samp * lamb_tilde[...,None]).transpose(0, 2, 1) @ phi_samp)/num_train_samples
    A_phi = ((x0_samp * lamb_tilde)[:,None,:] @ phi_samp)/num_train_samples
    Sig_phi = np.einsum('bnf,bng,bn->bfg', phi_samp, phi_samp, lamb_tilde, optimize=True)/num_train_samples
    print('Sig_phi shape: ', Sig_phi.shape)
    print('A_phi shape: ', A_phi.shape)
    print('computed empirical matrices')
    
    Sig_phi_inv = np.linalg.inv(Sig_phi + eps_reg*np.eye(Sig_phi.shape[-1])[None,:,:])
    print('inverted cov')
    
    W = A_phi @ Sig_phi_inv
    W = W.reshape(W.shape[0], W.shape[-1])
    print('computed W')
    return W, Sig_phi, A_phi.reshape(A_phi.shape[0], A_phi.shape[-1])



# simple implementation of sampling via first-order PF-ODE integration
# grid of pre-evaluated denoiser values used to avoid many function calls
def sample_PF_ODE_1d(num_samples, x, t, time_funcs, mu_est):
    num_x = len(x)
    b, g_sq, a, sig_sq = time_funcs['beta_t'], time_funcs['g_sq_t'], time_funcs['alpha_t'], time_funcs['sig_sq_t']

    x_t = np.random.normal(loc=0, scale=np.sqrt(sig_sq[0]), size=num_samples)  # Init state
    dt = np.roll(t,-1) - t; num_tsteps = len(t)-1         # get time steps

    inds = np.zeros((num_tsteps, num_samples),dtype=int)
    # Euler steps
    for j in range(num_tsteps):
        # Find index corresponding to current state
        i_ = np.clip(np.searchsorted(x, x_t), None, num_x-1); inds[j] = i_
        
        x_t = x_t - ( b[j]*x_t + (g_sq[j]/2)*((a[j]*mu_est[i_, j] - x_t )/sig_sq[j]))*dt[j] 

    return x_t


