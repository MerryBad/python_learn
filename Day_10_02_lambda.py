#
# def twice(n):
#     return n*2
# def proxy(f, n): # 콜백함수
#     return f(n)
# print(twice)
# print(twice(3))
#
# f=twice
# print(f(3))
#
# print(proxy(twice, 7))          # 함수를 매개변수로 전달 가능
# lam = lambda n:n*2              # 함수를 한줄로 표현하기
# print(lam(7))
# print(proxy(lam, 7))
# print(proxy(lambda n:n*2, 7))   # 코드를 매개변수로 전달 가능
#                                 # 가독성 + 효율성
# print('-'*50)

# 리스트를 오름차순/내림차순으로 정렬하기
# a = [5,1,9,3]
# a.sort()  # inplace 정렬
# b=sorted(a) # outplace 정렬
# print(sorted(a,reverse=False))
# # print(sorted(a)[::-1])
# print(sorted(a,reverse=True))
# for i in range(len(a)):
#     for j in range(i):
#         if a[i] < a[j]:
#             a[i], a[j] = a[j], a[i]
# print(a)
# for i in range(len(a)):
#     for j in range(i):
#         if a[i]>a[j]:
#             a[i], a[j] = a[j], a[i]
# print(a)

# colors를 오름차순, 내림차순으로 정렬
colors=['Red', 'green', 'blue', 'YELLOW']
# print(sorted(colors))
# print(sorted(colors, reverse=True))
#
# def make_lower(s):
#     print(s)
#     return s.lower()
#
# print(sorted(colors, key=make_lower))
# print(sorted(colors, key=str.lower))
# print(sorted(colors, key=lambda s:s.lower())) # key에 있는 값을 가져와서 정렬함

# colors를 길이순으로 정렬(내림차순)
print(sorted(colors, key=lambda s:len(s),reverse=True))
print(sorted(colors, key=lambda s:-len(s)))
print(sorted(colors, key=len))