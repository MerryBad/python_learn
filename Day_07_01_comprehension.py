import random
#
# a=[]
# for i in range(10):
#     if i % 2:
#         a.append(i)
#
# b= [i for i in range(10) if i % 2]
#
# print(a)
# print(b)
#
# a1=[random.randrange(100) for _ in range(10)]
# a2=[random.randrange(100) for _ in range(10)]
# a3=[random.randrange(100) for _ in range(10)]
# c=[a1,a2,a3]
# for i in c:
#     print (i)
# print ()
# 2차원 리스트의 합계를 구하기
# print (sum(a1)+sum(a2)+sum(a3))
# print ([sum(a1)+sum(a2)+sum(a3)])
# print (sum([sum(a1),sum(a2),sum(a3)]))
#
# print (sum([sum(i) for i in c]))

# 2차원 리스트를 1차원으로 만들기
# test=[j for i in c for j in i]
# for i in c:
#     for j in i:
#         test.append(j)
# print(test)

# 2차원 리스트에서 홀수만으로 1차원 리스트 만들기
# print([j for i in c for j in i if j%2])
# 2차원 리스트에서 가장 큰 숫자 찾기
# print(max([j for i in c for j in i]))

# 1~10000 사이에 들어있는 8의 갯수는?
# 808 -> 2
print([j for i in range(10000) for j in str(i) if j.count('8')].count('8'))
print(sum([str(i).count('8') for i in range(10000)]))

print(list(range(10)))
print(str(list(range(10))))
print(str(list(range(10000))).count('8'))