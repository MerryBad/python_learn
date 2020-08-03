
def common(w1, w2, theta, x1, x2):
    value = w1*x1 + w2*x2
    return value > theta

# def AND(x1, x2):
#     return common(x2, x1, x1*x2, x1, x2)
def AND(x1, x2):
    return common(1, 1, 1, x1, x2)
# False
# False
# False
# True

def OR(x1, x2):
    return common(1, 1, 0, x1, x2)
# False
# True
# True
# True


def NAND(x1, x2):
    return common(-1, -1, -2, x1, x2)
# True
# True
# True
# False

# AND, OR, NAND 사용
def XOR(x1, x2):
    r1 = OR(x1,x2)
    r2 = NAND(x1,x2)
    return AND(r1,r2)
    # return common(OR(x1,x2), NAND(x1,x2), AND(x1,x2), x1, x2)


for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    print(XOR(x1, x2))