
class Info:
    def __init__(self):
        print('init')
        self.age=12         # 멤버변수

    def show(self):         # 멤버함수
        print('show', self.age)

i1 = Info()
i2 = Info()
i2.addr='cheju' # 멤버변수를 밖에서 선언
print(i1)

i1.show()
Info.show(i1)
print(i1.age)
##
# a=[5,1,3,9]
# a.sort()
# list.sort(a)
# print(a)
##
