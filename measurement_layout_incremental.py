# %%
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import numpy.typing as npt
from pytensor.printing import Print
from particles import distributions as dists
from particles import state_space_models as ssm
# X is navigation, visual, bias, [distance, szie, behind, x_pos]
def logistic(x):
    return 1 / (1 + np.exp(-x))
class Measurement_Layout_AAIO(ssm.StateSpaceModel):
    default_params = {'sigmanav': 2.e-4,
                      'sigmavis': 1.e-4,
                      'sigmabias': 1.e-3,
                      'x0': np.array([1, 1, 1, 1, 1, 1, 1])
                     }

    def PX0(self):
        return dists.IndepProd(dists.Normal(loc=self.x0[0], scale = self.sigmanav),
                               dists.Normal(loc=self.x0[1], scale = self.sigmavis),
                               dists.Normal(loc=self.x0[2], scale = self.sigmabias),
                               )

    def PX(self, t, xp):
        return dists.IndepProd(dists.Normal(loc=xp[:, 0], scale = self.sigmanav),
                               dists.Normal(loc=xp[:, 1], scale = self.sigmavis),
                               dists.Normal(loc=xp[:, 2], scale = self.sigmaX),
                               )

    def PY(self, t, xp, x):
        rightlefteffect = x[:, 2] * x[:, 6]
        nav_performance = logistic(x[:, 0] - (x[:, 3] *(x[:, 5]*0.5+1.0) + rightlefteffect))
        vis_performance = logistic(x[:, 1] - np.log(x[:, 3]/x[:, 4]))
        final_prob = nav_performance * vis_performance
        return dists.Binomial(n=1, p = final_prob)
      
bear = Measurement_Layout_AAIO(sigmanav=5, sigmaX=5, sigmavis = 5, sigmabias=5)
x, y = bear.simulate(30)
xarr = np.array(x).squeeze()
plt.plot(y)
plt.show()