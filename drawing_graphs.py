import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def sigmoid(x): return 1 / (1 + np.exp(-(x-40)/10))
x = np.linspace(0, 100, 100)
y = np.ones(100)
plt.plot(x, y, color='black')
plt.show()