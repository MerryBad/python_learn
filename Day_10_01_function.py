
# def f_1(a, b, c):
#     print(a,b,c)

# f_1(1,2,3)          # positional
# f_1(a=1,b=2,c=3)    # keyword
# f_1(c=3,a=1,b=2)    # keyword
# f_1(1,2,c=3)        # positional+keyword
# f_1(1,b=2,3)      # SyntaxError: positional argument follows keyword argument
#                   # keyword는 positional 뒤에만

# def f_2(a=0,b=0,c=0):           # default argument
#     print(a, b, c, end='\n')
#
# f_2()
# f_2(1)
# f_2(1,2)
# f_2(1,b=2,c=3)
# f_2(c=3)
# f_2(b=2)

# def f_3(*args):             # 가변 인자
#     print(args, *args)      # unpacking
#
#
# f_3()
# f_3(1)
# f_3(1,'2')
# a=1,'2'                     # packing
# print(a)
#

def f_4(**kwargs):              # 키워드 가변 인자
    print(kwargs)               # 딕셔너리
    f_5(**kwargs)               # 실행하는 라인보다 앞에서 정의되어 있으면 에러 없음
    # f_5(k=kwargs)

def f_5(**kwargs):
    pass

f_4()
f_4(a=1, b=2)
f_4(ke='abc')
