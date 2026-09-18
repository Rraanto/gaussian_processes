"""
This script defines more generally the surrogate model of a 1D_function as in the notebook `notebook_test_GP.py`

The Surrogate class is made such that it encodes higher level modeling decisions (observation noise, ...) while the GP class is defined only with GP-dependent properties
"""

import gpytorch as gp
from gpytorch.mlls import marginal_log_likelihood
import torch 

class GP(gp.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean, covar):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean
        self.covar_module = covar 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gp.distributions.MultivariateNormal(mean_x, covar_x)

class Surrogate():
    def __init__(self, func, bounds, noise=False, sample_size=10, input_sample=None):
        """
        func: is a callable 1D function 
        bounds = is a tuple of the x bounds 
        sample_size defines how many points are sampled 
        if input_sample is given, sample_size is ignored and replaced by the size of input_sample
        """
        self._func = func

        self.bounds = bounds
        if input_sample is None: 
            x_min, x_max = self.bounds 
            ## generate uniformly distributed training sample in specified bounds 
            train_x = torch.rand(sample_size)
            train_x = (1 - train_x) * x_min + train_x * x_max
        else:
            train_x = input_sample

        self._train_x = train_x
        self._train_y = self._func(self._train_x)

        ## initialise likelihood 
        if not noise:
            self.likelihood = gp.likelihoods.FixedNoiseGaussianLikelihood(
                noise=torch.full_like(self._train_x, 1e-4)
            )
        else:
            self.likelihood = gp.likelihoods.GaussianLikelihood()


        ## Register a gp model as defined by the GP class
        self.gp_model = GP(
            train_x = self._train_x,
            train_y = self._train_y,
            likelihood = self.likelihood,
            mean = gp.means.ConstantMean(),
            covar = gp.kernels.ScaleKernel(gp.kernels.RBFKernel())
        )

        ## Switch to know whether model has trained 
        self.trained = False 

    def _check_trained(self):
        """
        Small util function that raises an error if not trained yet 
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet")

    def _set_train_mode(self):
        """
        sets to training mode 
        """
        self.likelihood.train()
        self.gp_model.train()

    def _set_eval_mode(self):
        """
        Sets to eval mode 
        """
        self.likelihood.eval()
        self.gp_model.eval()

    def get_training_data(self):
        return self._train_x, self._train_y

    def acquire_training_data(self, new_x):
        """
        Acquire new training data 

        training should only occur from the exterior even after acquiring new training data 
        """

        new_y = self._func(new_x)
        train_x_new = torch.cat([self._train_x, torch.tensor([new_x])])
        train_y_new = torch.cat([self._train_y, torch.tensor([new_y])])

        self._train_x = train_x_new 
        self._train_y = train_y_new 
        self.gp_model.set_train_data(inputs=self._train_x, targets=self._train_y, strict=False)

    def train(self, training_iters=50, verbose=False, details=False):
        """
        Train the internal gp

        if details = True a dictionary containing training information is returned 
        """
        self._set_train_mode()

        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.1)
        marginal_log_likelihood = gp.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.gp_model
        )
        
        loss_items = []

        ## main training loop 
        for i in range(training_iters):
            optimizer.zero_grad()
            output = self.gp_model(self._train_x)
            loss = -marginal_log_likelihood(output, self._train_y)
            loss.backward()

            if verbose:
                print('Iter %d/%d, loss: %.3f' % (i+1, training_iters, loss.item()))

            loss_items.append(loss.item())

            optimizer.step()

        self.trained = True ## indicate that model trained 

        if details:
            return {
                "loss": loss_items,
            }

    def __call__(self, x):
        """
        Making a prediction using the (trained GP model)

        Returns the mean, the lower and upper bound of the confidence region: 
        mean, lower, upper 
        """
        self._check_trained()
        self._set_eval_mode()

        with torch.no_grad(), gp.settings.fast_pred_var():
            output_distribution = self.likelihood(self.gp_model(x))

        ## return mean and unpacked confidence bounds 
        return output_distribution.mean, *output_distribution.confidence_region()

    def sample_output(self, x, N=10):
        """
        Sample from the output distribution at an input x 

        returns a N by d where x contains d input points
        """
        self._check_trained()
        self._set_eval_mode()
        
        output = self.likelihood(self.gp_model(x))
        samples = output.sample(torch.Size([N]))

        return samples

## unit test 
if __name__ == "__main__":
    from matplotlib import pyplot as plt 
    import numpy as np 

    def f(x):
        return torch.cos(torch.sin(x)) * torch.cos(x)

    bounds = (-5, 5)

    x = torch.linspace(*bounds, 100)
    y = f(x)

    plt.figure()
    plt.plot(x, y, "k--", label="Original function")

    gp_model = Surrogate(
        func = f,
        bounds=bounds,
        noise=False,
        sample_size=20,
        input_sample = torch.linspace(*bounds, 10)
    )

    train_results = gp_model.train(verbose=False, details=True)

    gp_y, lower, upper = gp_model(x)
    plt.plot(x, gp_y, label="GP mean Inference")
    plt.fill_between(x, lower, upper, alpha=0.3, label="GP confidence region")

    train_x, train_y = gp_model.get_training_data()
    plt.scatter(train_x, train_y, marker='o', label='Observed points', color="red")

    plt.xlabel("$x$")
    plt.ylabel("$y$")

    plt.title("GP regression of $x \\mapsto cos(sin(x)) cos(x)$")
    plt.legend()
