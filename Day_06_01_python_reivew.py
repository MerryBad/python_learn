# # Day_06_01_python_review.py
#
# # 문제
# # 0~9까지 리스트 만들고 거꾸로 출력하기
#
# list_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# print(list_num[::-1])
# for i in reversed(list_num):
#     print(i, end=" ")
# print()
# ########################################
# import copy
# # 앞에서 만든 리스트를 다른 리스트에 깊은 복사하기
# list_copy = []
# list_copy = copy.deepcopy(list_num)
# print(list_copy)
# ##########################################
# temp=tuple(list_num)
# list_test=list(list_num)
# print(list_test)
# #########################################
# list_cop=list_num[:]
# print(id(list_num))
# print(id(list_cop))
# print(list_cop)
########################################
# d=[]
# for i in list_num:
#     d.append(i)
# print(d)
#########################################

# 두 개의 숫자 중에서 큰 숫자를 찾는 함수 만들기
#
# def max2(a,b):
#     if a<b:
#         a=b
#     return a
#
# print(max2(10,20))
# print(max2(20,10))

# 네 개의 숫자 중에서 큰 숫자를 찾는 함수를 만드세요

# def max4(a,b,c,d):
#     test=[a,b,c,d]
#     max_num=0
#     for i in test:
#         if i>max_num:
#             max_num=i
#     return max_num
#
# def max4(a,b,c,d):
#     # if a < b: a = b
#     # if a < c: a = c
#     # if a < d: a = d
#     # return a
#     # 복면가왕-
#     # return max2(max2(a,b),max2(c,d))
#     # 한국시리즈-
#     return max2(max2(max2(a,b),c),d)
# print(max4(1,2,3,4))
# print(max4(1,4,2,3))
# print(max4(4,2,1,3))

#############################################

# #
# for i in range(5):
#     i
# # 컬렉션을 만드는 한 줄짜리 반복문
# [i for i in range(5)]
# (i for i in range(5))
# {i for i in range(5)}
#
# print([i for i in range(5)])
# print(sum([i for i in range(5)]))
#
import random
#
# for _ in range(10):
#     print(random.randrange(100),end=' ')
# print()
#
# # 100보다 작은 난수가 10개 들어있는 리스트 만들기
test=[random.randrange(100) for _ in range(10)]
print(test)

# 리스트에서 홀수만 뽑아서 리스트 만들기
odd=[test[i] for i in range(len(test)) if test[i]%2]
print(odd)

for i in test:
    if i %2:
        print(i, end=" ")
print([i for i in test if i%2])