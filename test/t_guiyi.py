def Normalization(x):
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]


x = [0.6, 0.7, 0.9]

print(Normalization(x))
