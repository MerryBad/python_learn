import numpy as np
import matplotlib.pyplot as plt
def sigmoid(z):
    return 1/(1+np.e**-z)

print(np.e)
print('-'*30)
print(sigmoid(-1))
print(sigmoid(0))
print(sigmoid(1))
print('-'*30)

# -10~10 사이의 결과에 대해 시그모이드 그래프 그리기

x = np.arange(-10,11,0.2)
plt.plot(x,sigmoid(x),'ro')
plt.show()