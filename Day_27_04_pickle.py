import pickle
def save():
    d = {'age':23, 'name':'kim'}
    f=open('data/dict.pkl', 'wb')
    pickle.dump(d, f)
    f.close()
def load():
    f=open('data/dict.pkl', 'rb')
    d = pickle.load(f)
    print(d)
    f.close()
save()
load()