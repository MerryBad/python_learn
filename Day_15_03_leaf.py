import pandas as pd
import numpy as np
from sklearn import svm, model_selection, preprocessing

#leaf.csv 파일을 읽어서 70%의 데이터로 학습하고 30% 데이터에 대해 예측하기

# 1. 팬더스로 파일 읽기
# 2. x와 y 데이터 만들기
# 3. 학습과 검사 데이터 셋으로 분할
# 4. 학습하기
# 5. 예측해서 정확도 계산하기

# 1. 팬더스로 파일 읽기
leaf = pd.read_csv('data\leaf.csv')
# print (leaf)


# 2. x와 y 데이터 만들기
le = preprocessing.LabelEncoder()
le.fit(leaf.species)
data = leaf.values
x = data[:, 2:]
y = le.transform(leaf.species)
print(x.shape,y.shape)
print(y[:5])


# 3. 학습과 검사 데이터 셋으로 분할
data=model_selection.train_test_split(x,y,train_size=0.7)
x_train, x_test, y_train, y_test = data


# 4. 학습하기
clf = svm.SVC()
clf.fit(x_train, y_train)


# 5. 예측해서 정확도 계산하기
y_hats=clf.predict(x_test)
equals=(y_hats==y_test)
print('acc : ', np.mean(equals))
# #------------------------------------#
# clf = svm.SVC(gamma=0.001, C=2)
# clf.fit(x_train, y_train)
#
# y_hats=clf.predict(x_test)
# equals=(y_hats==y_test)
# print('acc : ', np.mean(equals))