from matplotlib import pyplot as plt
from matplotlib import pylab
import numpy as np

x = np.linspace(-10,10,100)
y1 = np.log(1+np.exp(-x))
y2 = np.maximum(0,-x)

plt.plot(x,y1,"r-",label="log(1+exp(-x))")
plt.plot(x,y2,"b-",label="max(0,-x)")

plt.legend()
pylab.savefig("approx.png")
