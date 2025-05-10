import numpy as np

        
def A_plus(w, wMin, wMax, beta_pre): #pre-synaptic
    return np.exp(-beta_pre * (wMax - w) / (wMax - wMin))
    
def A_minus(w, wMin, wMax, beta_post): #post-synaptic
    return np.exp(-beta_post * (w - wMin) / (wMax - wMin))
    
def x(dt, tNorm, gamma_pre): #pre-synaptic
    return 0.5*(1+np.sign(dt))*np.abs(dt/tNorm)*np.exp(-gamma_pre * np.square((dt/tNorm)))
    
def y(dt, tNorm, gamma_post): #post-synaptic
    return 0.5*(1-np.sign(dt))*np.abs(dt/tNorm)*np.exp(-gamma_post * np.square((dt/tNorm)))
    
def delta_w(dt, w, wMin, wMax, tNorm, alpha_pre, alpha_post, beta_pre, beta_post, gamma_pre, gamma_post):
    return alpha_pre * A_plus(w, wMin, wMax, beta_pre) * x(dt, tNorm, gamma_pre) - alpha_post * A_minus(w, wMin, wMax, beta_post) * y(dt, tNorm, gamma_post)

# Map names in NEST to Davydov's.
parameter_name_mapping = {
    'weight': 'w',
    'delta_t': 'dt',
    'tau_plus': 'tNorm',
    'tau_minus': 'tNorm',
    'Wmax': 'wMax',
    'Wmin': 'wMin',
    'alpha_plus': 'alpha_pre',
    'alpha_minus': 'alpha_post',
    'beta_plus': 'beta_pre',
    'beta_minus': 'beta_post',
    'gamma_plus': 'gamma_pre',
    'gamma_minus': 'gamma_post',
}

default_synapse_parameters = {
    'wMax': 1,
    'wMin': 0,
    'tNorm': 10,
    'alpha_pre': 0.3161977426412055,
    'alpha_post': 0.011297914920262086,
    'beta_pre': 2.213043124548639,
    'beta_post': -5.9693974750170025,
    'gamma_pre': 0.031839429846118886,
    'gamma_post': 0.14568323501051358,
}

def get_delta_G_from_fit(**parameters_in_nest_format):
    '''A wrapper to obtain a conductance change
    from Davydov's fit.'''
    parameters_in_davydovs_format = {
        parameter_name_mapping[name]: value
        for name, value in parameters_in_nest_format.items()
        if name in parameter_name_mapping
    }
    # Set missing parameters to default.
    all_parameters_in_davydovs_format = default_synapse_parameters
    all_parameters_in_davydovs_format.update(parameters_in_davydovs_format)
    return delta_w(**all_parameters_in_davydovs_format)
