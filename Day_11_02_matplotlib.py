import numpy as np
import matplotlib.pyplot as plt
def plot_1():
    # plt.plot([1,3,5,7])
    # plt.plot([1,3,5,7], [1, 2, 3, 5])
    # plt.plot([1,3,5,7], [1, 2, 3, 5], 'g') # 꺾은 선 그래프
    # plt.plot([1,3,5,7], [1, 2, 3, 5], 'o') # 산점도
    # plt.plot([1,3,5,7], [1, 2, 3, 5], 'x')
    plt.plot([1,3,5,7], [1, 2, 3, 5], 'gㅌ')
    plt.show()
# y=x ^ 2를 그래프로 그리기
def plot_2():
    # x=np.arange(-10,11)
    # y=[i**2 for i in x]
    # plt.plot(x, y)
    x=np.arange(-5,5, 0.1)
    plt.plot(x,x**2,'rx')
    plt.plot(x,x**2)
    plt.show()
# 4개의 로그 그래프를 하나의 플랫에 그리기
def plot_3():
    x=np.arange(-10,11,0.1)
    y=np.log(x)
    plt.plot(x,y, 'r')
    plt.plot(x,-y, 'g')
    plt.plot(-x,y, 'b')
    plt.plot(-x,-y, 'orange')
    plt.show()
def plot_4():
    x=np.arange(-1,2,0.1)
    y=np.log(x)
    plt.subplot(2,2,1)
    plt.plot(x,y, 'r')
    plt.subplot(2,2,2)
    plt.plot(x,-y, 'g')
    plt.subplot(2,2,3)
    plt.plot(-x,y, 'b')
    plt.subplot(2,2,4)
    plt.plot(-x,-y, 'orange')
    plt.show()
def plot_5():
    x=np.arange(-1,2,0.1)
    y=np.log(x)

    plt.figure(1)
    plt.plot(x,y, 'r')
    plt.figure(2)
    plt.plot(x,-y, 'g')
    plt.figure(2)
    plt.plot(-x,y, 'b')
    plt.figure(1)
    plt.plot(-x,-y, 'orange')
    plt.show()

# plot_1()
# plot_2()
# plot_3()
# plot_4()
plot_5()