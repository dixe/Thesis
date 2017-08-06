import cv2
import numpy as np
import matplotlib.pyplot as plt

def decay(x):
    return np.power(np.e, -x)

def val(x):

    x2 = (x-4)
    return 0.025 + ( np.power(np.e, -x) if x <= 4 else np.power(np.e, -4) + 0.001 * np.power(x2,2))

x = np.arange(1,10,0.1)

y = [decay(i) for i in x]

yVal = [val(i) for i in x]

y = y + np.random.normal(0,0.003,len(y))
yVal = yVal + np.random.normal(0,0.003,len(y))

plt.plot(x,y,color='k', label="Training")
plt.plot(x,yVal, color='k', linestyle="dashed", label="Validation")
plt.legend()

plt.show()
