# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# This notebook tests how to use `gpytorch` for Gaussian Process modeling. 

# %%
# %load_ext autoreload 
# %autoreload 2

# %%
import numpy as np 
import gpytorch
from matplotlib import pyplot as plt 
import torch


# %% [markdown]
# # 1D model 
#
# First, a GP is used to approximate a function from $\mathbb R \rightarrow \mathbb R$ using a small amount of sample points $(N=5)$, and using a fairly simple Kernel. This part is an (almost) exact reproduction of a [tutorial from `gpytorch` documentation](https://github.com/cornellius-gp/gpytorch/blob/main/examples/01_Exact_GPs/Simple_GP_Regression.ipynb), where random noise is added on the sampled values. Namely, the model whose outputs are considered unknown is the function: 
#
# $$
# f: 
# \begin{cases}
# [0, 1] \longrightarrow \mathbb R \\ 
# x \longmapsto sin(2\pi x)
# \end{cases}
# $$
#
# An initial dataset $\{ (x_1, f(x_1)), \dots, (x_N, f(x_N))\}$ is generated where $(x_i)_{i \leq N}$ comes from a uniform distribution in $[0, 1]$. 

# %%
## 1D example function 

def f(x):
    return torch.sin(x * (2 * torch.pi))

x = torch.linspace(0, 1, 100) 
y = f(x)

plt.figure() 
plt.plot(x, y, "k-")

# %%
## Sample points along the curve: 

N = 5 

train_input_sample = torch.rand(N)

plt.figure()
plt.plot(x, y, 'b--')
plt.scatter(train_input_sample, f(train_input_sample), color='red', marker='o')


# %% [markdown]
# # Using GPyTorch 
#
# ## A class for the GP surrogate 
#
# The philosophy in `GPyTorch` is that an user can create classes that will represent the surrogate models using building blocks that are provided (mean functions, kernels, prior distributions). The `GP` class defined in the next cell is a minimum such class whose associated GP model can later be conditionned (trained) on the sample and used for inference. 
#
# The base class that should be used as a template is `gpytorch.models.ExactGP`. It contains the attributes: 
#
# - `mean_module`: The mean function, many ready-to-use instances are provided by `gpytorch.means`.
# - `covar_module`: The covariance function, classical examples are provided by `gpytorch.kernels`.

# %%
class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean, covar):
        """
        train_x: the training input set 
        train_y: the training output set 
        likelihood: The likelihood function 
        mean: the mean 
        covar: the covariance 
        """
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean 
        self.covar_module = covar 

    def forward(self, x):

        ## These characterise the output Normal Distribution 
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

## The likelihood: 
## In the no noise setup, a fixed very small noise likelihood is used 
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
    noise=torch.full_like(x, 1e-6)
)

train_x, train_y = train_input_sample, f(train_input_sample)

## This is the GP model associated to f: 
model = GP(
    train_x = train_x,
    train_y = train_y,
    likelihood = likelihood,
    mean = gpytorch.means.ConstantMean(),
    covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
)

# %% [markdown]
# ## Training the GP surrogate 
#
# This step is (equivalent) to conditionning the model to the dataset that has been assigned. Since `GPyTorch` internally uses `pytorch`, training the defined GP model is exactly similar to training a pytorch model (in fact, the gp models in `gpytorch.models` extend `torch.nn.Module` as per [The original example from `GPyTorch`](https://notebooks.githubusercontent.com/view/ipynb?browser=firefox&bypass_fastly=true&color_mode=dark&commit=6272eda426c7d4115b0911efcf053a37ab956af9&device=unknown_device&docs_host=https%3A%2F%2Fdocs.github.com&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f636f726e656c6c6975732d67702f677079746f7263682f363237326564613432366337643431313562303931316566636630353361333761623935366166392f6578616d706c65732f30315f45786163745f4750732f53696d706c655f47505f52656772657373696f6e2e6970796e62&link_underline_enabled=true&logged_in=true&nwo=cornellius-gp%2Fgpytorch&path=examples%2F01_Exact_GPs%2FSimple_GP_Regression.ipynb&platform=linux&repository_id=93868719&repository_type=Repository&version=145#fcbd28dc-e0aa-44fe-9f84-4b9242e0e183). 

# %%
## Training setup 
training_iters = 50

## set to train mode 
model.train()
likelihood.train() 

## choose optimiser 
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

## The GP should minimise the mll 
marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iters):
    optimizer.zero_grad()
    output = model(train_x)

    ## loss function update 
    loss = -marginal_log_likelihood(output, train_y)
    loss.backward() 

    print('Iter %d/%d, loss (-mll): %.3f' % (i+1, training_iters, loss.item()))

    optimizer.step()

# %% [markdown]
# ## Inference 
#
# Once the GP is conditionned, the model is used to infer the values of $f$ at unseen $x$. In this particular experiment setup ((almost) zero noise), the prediction provides interpolations of the sample points. 

# %%
## evaluation mode 
model.eval()
likelihood.eval() 

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 100)

    ## The GP output is a function of normal distribution (posterior) per x 
    model_output = likelihood(model(test_x))

# %%
## sample curves of the posterior
sample_curves = model_output.sample(torch.Size([10]))

plt.figure()
plt.plot(test_x, f(test_x), "b-")
for curve in sample_curves:
    plt.plot(test_x, curve, color="grey", linestyle="-", alpha=0.3)

# %%
with torch.no_grad():
    ## dissect the output distribution 
    output_mean = model_output.mean.numpy() 
    lower, upper = model_output.confidence_region()

    ## plot prediction, known function and sampled training data 
    plt.figure() 
    plt.plot(test_x, f(test_x), color="grey", linestyle="--", label="True function")
    plt.scatter(train_x, train_y, color="red", marker="o", label="Training points") 
    plt.plot(test_x, output_mean, "b-", label="Predicted mean")
    plt.fill_between(test_x, lower, upper, color="blue", alpha=0.1, label="Confidence region")
    plt.legend()
