a = [1, 2, 3]


def t1():
    for i in range(len(a)):
        yield a[i], i


for ii in  t1():
    print(ii[0])

