import numpy as np  


def fake_capacitance_model(n_steps, max_steps, cgd, alpha = 0.15, beta = 0.05):

    error = np.random.normal(0, beta + alpha*(1-n_steps/max_steps), (cgd.shape[0], cgd.shape[1]))

    pred_cgd = np.clip(cgd + error, 0, 1)

    return pred_cgd