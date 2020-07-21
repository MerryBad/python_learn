import numpy as np

# print(np.zeros(3))
# print(np.ones(3))
# print(np.full(3, fill_value=-1))
# print(np.zeros([2,3]))
# print(np.zeros([2,3], dtype=np.int32))

# print(np.random.random(5))
# print(np.random.random([2,3]))
# print(np.random.randn(5))
# # print(np.random.randn([2,3]) # error
# print(np.random.randn(6).reshape(2, 3)) # 정규분포
# print(np.random.uniform(0,10,5))
# print(np.random.random_sample(5))
# print(np.random.choice(range(10), 15))

# np.random.seed(23)
# a=np.random.choice(range(10), 15).reshape(3,5)
# print(a)
#
# print(np.sum(a))
# print(np.sum(a, axis=0)) # 열(수직)
# print(np.sum(a, axis=1)) # 행(수평)
#
# # 2차원 배열에서 가장 큰 값과 작은 값 찾기
# print(a.max())
# print(a.min())
# print(np.max(a, axis=0))
# print(np.max(a, axis=1))

# print(np.mean(a))

# print(np.argmax(a)) # arg = 인덱스
# print(np.argmax(a, axis=0)) # arg = 인덱스
# print(np.argmax(a, axis=1)) # arg = 인덱스

# print(a[0])
# print(a[-1])

# a[0]=-1
# print(a)

# 2차원 배열을 거꾸로 출력
# print(a[::-1])
# # print(a[::-1][::-1]) # => b=a[::-1], c=b[::-1]
# print(a[::-1, ::-1]) # fancy indexing

# 2차원 배열의 첫번째와 마지막번째의 값을 -1로 바꾸기
# a[0,0]=-1
# a[-1,-1]=-1
# print(a)
#
# b=a[0]
# b[2]=-3
# print(a)

# 속은 0이고 테두리가 1로 채워진 5행 5열 배열
# b=np.zeros([5,5], dtype= np.int32)
# b[0]=1
# # b[0,:]=1
# b[-1]=1
# b[:,0]=1
# b[:,-1]=1
# print(b)
#
# c=np.ones([5,5], dtype= np.int32)
# # c[1:4,1:4]=0
# c[1:-1,1:-1]=0
# print(c)


#2차원 배열을 열 단위로 출력하기
# e=np.arange(20).reshape(4,5)
# print(e)

# print(np.transpose(e))

# print(e.size) # 20
# print(e[:,0].size) # 4
# print(e.shape[0]) # 4
# print(e[0].size) # 5
# print(e.shape[1]) # 5
# for i in range(e.size//e[:,0].size): # C언어가 뇌를 지배..
#     for j in range(e[:,0].size):
#         print(e[j,i], end=" ")
#     print()

# for i in range(e.shape[1]):
#     for j in range(e.shape[0]):
#         print(e[j, i], end=' ')
#     print()

# for i in range(e.shape[0]):
#     for j in range(e.shape[1]):
#         print('({}, {}) {}'.format(i,j,e[i,j]), end=" ")
#     print()

# for i in range(e.shape[1]):
#     print(e[:,i])

# f=np.arange(10)
# print(f)
# print(f[0], f[1])
# print(f[[0,1]]) # list 해야됨
#
# g=[1,5,3,2,3]
# print(f[g])     # index 배열

# h=f.reshape(2,5)
# print(h)
# print(h[0],h[1])
# print(h[[0,1]])
# print(h[[0,1,1,0]])

# 단위행렬(대각선이 1로 채워진 행렬)
# 5행 5열의 단위행렬을 만들기

print(np.eye(5, dtype=np.int32))

r=np.zeros([5,5], dtype=np.int32)
# index=[0,1,2,3,4]
# r[(index,index)]=1
r[range(5),range(5)]=1
print(r)

for i in range(len(r)):
    r[i,i]=1