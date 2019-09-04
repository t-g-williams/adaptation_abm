import numpy as np

class Climate():
    def __init__(self, inputs):
        # attribute the parameters to the object
        self.all_inputs = inputs
        self.inputs = inputs['climate']
        for key, val in self.inputs.items():
            setattr(self, key, val)
        self.T = self.all_inputs['model']['T']

        # create the entire sequence of climate realizations
        # [self.rain_alpha, self.rain_beta] = find_beta_params(self.rain_mu, self.rain_var)
        # self.rain = np.random.beta(self.rain_alpha, self.rain_beta, self.T)
        self.rain = np.random.normal(self.rain_mu, self.rain_sd, self.T)
        self.rain[self.rain < 0] = 0
        self.rain[self.rain > 1] = 1

# def find_beta_params(mu, var):
#     '''
#     find the beta distribution parameters (alpha and beta)
#     for the distribution with the given mean and variance
#     https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance
#     '''
#     alpha = mu**2 * ((1-mu)/var - 1/mu)
#     beta = alpha * (1/mu - 1)
#     return alpha, beta

