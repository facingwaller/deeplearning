a = [1, 2, 3]
import  math
def t2 ():
    a_neg = [1,2,3,4,5,6,7,8,9,10]
    batch_size = 3
    length = len(a_neg)
    geshu = math.ceil(length / batch_size)

    for i in range(geshu):
        yield a_neg[i*batch_size:(i + 1) * batch_size]

def t1():
    for i in range(len(a)):
        yield a[i], i


for ii in  t2():
    print(ii)

