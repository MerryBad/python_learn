from sklearn import datasets, svm

iris = datasets.load_iris()

clf=svm.SVC()
clf.fit(iris['data'], iris['target'])

print(clf.score(iris['data'], iris['target']))