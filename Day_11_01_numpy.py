import numpy as np
#                #시작 종료 증가
# print (np.arange(0, 10 ,1))
# print (np.arange(0, 10))
# print (np.arange(10))
# print (np.arange(-5, 5, 1))
# print (np.arange(0, 1, 0.2))
#
# # 배열 : 같은 공간, 같은 자료형
# print (type(np.arange(10)))

# a = np.arange(6)               # 행렬, 크기, 차원
# print(a.shape, a.size, a.ndim) # (6,)   6    1
#
# # b = a.reshape(2,3)
# # b = np.reshape(a, [2, 3])
# b = np.reshape(a, (2, 3))
# print(b)
# print(b.shape, b.size, b.ndim) # (2, 3) 6    2
# print(b.dtype)

# # 2차원 배열을 1차원으로 변환하기
# a = np.arange(6)
# b = a.reshape(2,3)
# c1 = b.reshape(6)
# c4 = b.reshape(len(a))
# c5 = b.reshape(b.size)
# c6 = b.reshape(b.shape[0]*b.shape[1])
# c7 = b.reshape(-1)        # 자동완성
# c2 = np.reshape(b, 6)
# c3 = np.reshape(b,(6,))
# print(a)
# print(b)
# print(c1)
# print(c2)
# print(c3)
# print(c4)
# print(c5)
# print(c6)
# print(c7)

# print(np.int32) # 8 16 32 64

# 1차원 배열을 2차원으로 변환하기
# a = np.arange(6)
# print(a.reshape(2,3))
# print(a.reshape(2,-1))
# print(a.reshape(-1,3))


# print(list(range(6))) # [0, 1, 2, 3, 4, 5] 쉼표가 있음
# print(np.array(range(6)))
# print(np.arange(6))
# print(np.arange(6).dtype)
# print(np.arange(6, dtype=np.int8).dtype)
# print(np.int64(range(6)))


# g = np.arange(6)
# h = np.arange(6)    # [0 1 2 3 4 5]
# # broadcast 배열 전체에 연산
# print(g+1)          # [1 2 3 4 5 6]
# print(g**2)         # [ 0  1  4  9 16 25]
# print(g>2)          # [False False False  True  True  True]
# print(type(g>2))
# print((g>2).dtype)
# # vector 연산
# print(g+h)          #[ 0  2  4  6  8 10]
# print(g**h)         #[   1    1    4   27  256 3125]
# print(g>h)
#
# print ( '-' * 50)
#
# i = g.reshape(2,3)
# j = h.reshape(2,3)
#
# print(i + 1)
# print(i ** 2)
# print(i > 2)
#
# print(i+j)
# print(i > j)
#
# print(np.sin(h))        # universal function
# print(np.sin(i))
#
# print ( '-' * 50)

a1=np.arange(3)
a2=np.arange(6)
a3=np.arange(3).reshape(1,3)
a4=np.arange(3).reshape(3,1)
a5=np.arange(6).reshape(2,3)
#
# # print(a1+a2) 다른 크기는 연산 안됨
# print(a1+a3) # 차원이 달라도 크기가 같으면 연산 됨 -> 에러 발생 가능
#              # (3,) => (1,3)
# print(a1+a4) # broadcast + broadcast 한쪽이 1일때
# print(a1+a5) # broadcast + vector    양쪽에 같은 숫자가 있을때

# a2, a3, a4, a5에 대해서 연산 해보기
# print(a2) # (1,6)
# print(a3) # (1,3)
# print(a2+a3) # 크기 다름
# print(a2) # (1,6)
# print(a4) # (3,1)
# print(a2+a4) # broadcast
# print(a2) # (1,6)
# print(a5) # (2,3)
# print(a2+a5) # 크기 다름
# print(a3) # (1,3)
# print(a4) # (3,1))
# print(a3+a4) # broadcast
# print(a3) # (1,3)
# print(a5) # (2,3)
# print(a3+a5) # vector
# print(a4) # (3,1)
# print(a5) # (2,3)
# print(a4+a5) # 크기 다름