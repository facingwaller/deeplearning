a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = [2, 5, 8, 11, 0]
c = [2, 5, 8, 11, 0]
# intersection
intersection = list(set(a).intersection(set(b)))
print(intersection)
# # union
# union = b.extend([v for v in a])
# # difference
# difference = [v for v in a if v not in b]