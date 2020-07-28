# import tensorflow as tf # 1.14.0
import matplotlib.pyplot as plt

def cost(x, y, w):
    c=0
    for i in range(len(x)):
        hx= w * x[i]
        loss = (hx-y[i])**2 # mse
        c += loss
    return c/len(x)

# hx = wx + b
#      1    0
#  y = ax + b
#  y = x

x=[1,2,3]
y=[1,2,3]

# print(cost(x,y,w=0))
# print(cost(x,y,w=1))
# print(cost(x,y,w=2))

# w가 일정 기간에 대해 변할 때의 loss를 그래프로 출력하기
def show_cost():
    for i in range(-30,50):
        w=i/10
        c=cost(x,y,w=w)
        print(w, c)
        plt.plot(w,c, "ro")

    plt.show()

# 미분 : 기울기, x가 y에 미치는 영향
#       x가 1씩 바뀔 때 y가 바뀌는 정도
#
# y = 3        3=1 3=2 3=3     0
# y = x        1=1 2=2 3=3     1
# y = 2x       2=1 4=2 6=3     2
# y = (x+1)    2=1 3=2 4=3     1
# y = 2(x+1)   4=1 6=2 8=3     2

# y = x^2      1=1 4=2 9=3     2x
# y = (x+1)^2                  2x


def gradient_descent(x,y,w):
    c=0
    for i in range(len(x)):
        hx= w * x[i]
        loss = (hx-y[i])*x[i]
        c += loss
    return c/len(x)

def show_gradient():
    x=[1,2,3]
    y=[1,2,3]

    # w를 1.0으로 만드는 코드를 찾기
    w=5

    for i in range(10):
        c=cost(x,y,w=w)  # 코스트 = 정답과 측정값의 차이
        g=gradient_descent(x,y,w=w)
        w-= 0.1* g
        print(i, c)

    # 반복횟수 늘리기
    # for i in range(1000):
    #     g = gradient_descent(x, y, w=w)
    #     w -= 0.1 * g
    #     print(i, w)

    # 값 바꾸기
    # for i in range(10):
    #     c=cost(x,y,w=w)
    #     g=gradient_descent(x,y,w=w)
    #     w-= 0.195* g
    #     print(i, c)

    # x가 5와 7일 때의 결과를 출력
    print('5 : ', w*5)
    print('7 : ', w*7)
show_gradient()