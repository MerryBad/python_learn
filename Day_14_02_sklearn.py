from sklearn import datasets, svm, model_selection
import numpy as np

def predict_1():
    iris = datasets.load_iris()

    clf=svm.SVC()
    clf.fit(iris['data'], iris['target'])

    y_hats=clf.predict(iris['data'])
    equals=(y_hats==iris['target'])
    print('acc : ', np.mean(equals))
#digits 데이터셋에서 마지막 데이터를 제외한 데이터로 학습하고
#마지막 데이터에 대해 결과를 예측하기
def predict_2():
    digits = datasets.load_digits()

    clf = svm.SVC()
    clf.fit(digits['data'][:-1], digits['target'][:-1])

    y_hats = clf.predict(digits['data'][-1:])
    equals = (y_hats == digits['target'][-1:]) # 2차원으로 만들어줌
    print(digits['target'][-1:], y_hats)
# 70% 데이터로 학습, 30% 데이터에 대해 정확도 예측
def predict_3():
    digits = datasets.load_digits()
    x=digits['data']
    y=digits['target']
    train_size=int(len(x)*0.7)
    # x_train, x_test = x[:train_size], x[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

    # 75 : 25
    # data=model_selection.train_test_split(x,y)
    # 70 : 30
    # data=model_selection.train_test_split(x,y,train_size=0.7)
    # 셔플 x, 동영상같은 시간에 따라 진행되는 것들은 셔플하면 안 됨
    data=model_selection.train_test_split(x,y,shuffle=False)
    x_train, x_test, y_train, y_test = data
    clf = svm.SVC()
    clf.fit(x_train, y_train)

    y_hats=clf.predict(x_test)
    equals=(y_hats==y_test)
    print('acc : ', np.mean(equals))
    #------------------------------------#
    clf = svm.SVC(gamma=0.001, C=2)
    clf.fit(x_train, y_train)

    y_hats=clf.predict(x_test)
    equals=(y_hats==y_test)
    print('acc : ', np.mean(equals))

# predict_1()
# predict_2()
predict_3()
