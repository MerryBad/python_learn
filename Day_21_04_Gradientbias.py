# import tensorflow as tf # 1.14.0
import matplotlib.pyplot as plt

x=[1,2,3]
y=[1,2,3]

def cost(x, y, w, b):
    c=0
    for i in range(len(x)):
        hx= w * x[i] + b
        loss = (hx-y[i])**2 # mse
        c += loss
    return c/len(x)



def gradient_descent(x, y, w, b):
    grad0,grad1=0, 0
    for i in range(len(x)):
        hx= w * x[i] + b
        grad0 += (hx-y[i])*x[i]
        grad1 += (hx-y[i])
    return grad0/len(x), grad1/len(x)

def show_gradient_bias():
    x=[1,2,3]
    y=[1,2,3]

    # w를 1.0으로 만드는 코드를 찾기
    w=5
    b=-3

    for i in range(1000):
        c=cost(x,y,w=w, b=b)  # 코스트 = 정답과 측정값의 차이
        g0, g1=gradient_descent(x,y,w=w, b=b)
        w-= 0.1* g0
        b-= 0.1* g1
        print(i, c)


    # x가 5와 7일 때의 결과를 출력
    print('5 : ', w*5+b)
    print('7 : ', w*7+b)
show_gradient_bias()