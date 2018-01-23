import numpy as np
def cap_nums( y, rate=0.8):
    y = y.copy()
    y = np.array(y)
    s = 0
    total_len = len(y)
    total_index = total_len * rate
    e = int(total_index)

    reverseIndex = int(total_len - total_index)
    # print(reverseIndex)
    # 正向截取
    # 逆向
    y1 = y[s:e]  # [ > s and <= e  ]
    # print(type(y1))

    y2 = y[-reverseIndex:]

    # print(total_len)
    print("split into 2 " + str(len(y1)) + " " + str(len(y2)))
    return y1, y2


a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b, c = cap_nums(a, 0.8)
print(b)
print(c)