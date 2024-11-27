# %%
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import numpy.typing as npt
from pytensor.printing import Print
from particles import distributions as dists


prior_dict = {'mu': dists.Normal(scale=2.),
              'rho': dists.Uniform(a=-1., b=1.),
              'sigma':dists.Gamma()}
my_prior = dists.StructDist(prior_dict)

theta = my_prior.rvs(size=500)  # sample 500 theta-parameters

plt.style.use('ggplot')
plt.hist(theta['sigma'], 30)
plt.xlabel('sigma')
plt.show()

plt.figure()
z = my_prior.logpdf(theta)
plt.hist(z, 30)
plt.xlabel('log-pdf')
plt.show()