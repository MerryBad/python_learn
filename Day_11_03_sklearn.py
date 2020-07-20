from sklearn import datasets
def basic_1():
    # Petal = 구분할 수 있는 특징
    iris=datasets.load_iris()
    print(type(iris))
    print(iris.keys()) # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
    print(iris['feature_names']) #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    print(iris['target_names']) #['setosa' 'versicolor' 'virginica']
    print(iris['data'][:5])
    # [[5.1 3.5 1.4 0.2]
    #  [4.9 3.  1.4 0.2]
    #  [4.7 3.2 1.3 0.2]
    #  [4.6 3.1 1.5 0.2]
    #  [5.  3.6 1.4 0.2]]
    print(iris['target']) # [0 0 0 ... 2 2]
    print(iris['target'].shape)
    print(iris['frame']) # None
    print(iris['DESCR'])
# digit 데이터 셋에 대해 알아보고 x 데이터의 첫 번째 데이터를 정확하게 출력하기
def basic_2():
    digits=datasets.load_digits()
    # print(digits.keys())
    # dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])

    # print(type(digits['data']))
    # <class 'numpy.ndarray'>

    print(digits['data'][:1])
    # print(digits['data'][:1].shape) # (1, 64)
    print(digits['images'][:1])
    #     print(digits['images'][:1].shape) # (1, 8, 8)
    print(digits['data'][:1].reshape(8,8))
# basic_1()
basic_2()