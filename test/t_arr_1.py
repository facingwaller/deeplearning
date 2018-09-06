import numpy as np

r1 = 'sadas21312萨达'
r2 = [x for x in r1]

print(r2)

_tmp_y = 'sadas21312萨达'
_1 = _tmp_y[0]
_tmp_y = list(_tmp_y)
del _tmp_y[0]
_tmp_y.append(_1)

print(_tmp_y)



arr = []

for i in range(100):
    a1 = []
    for k in range(100):
        a1.append('%d-%d' % (i, k))
    arr.append(a1)
arr = np.array(arr)

n = 0
n_steps = 100
x = arr[:, n:n + n_steps]
y = np.zeros_like(x)

print(x)
print(y)
y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]

print(x)
print(y)
