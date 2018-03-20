for i in range(0):
    print(i)

print(111111111)


def get_score_for_sort(question, sp):
    # 实体在句子中 1000分 【必须在，否则不考虑？】
    # 完全属性在句子中100分
    # 部分属性在句子中10分
    score = 0
    if str(question).__contains__(sp[0]):
        score += 100
    if str(question).__contains__(sp[1]):
        score += 10
    return score


s1 = set()

s1.add('a')

for s in s1:
    print(s)

vs = set()
t1 = ('a', 'b')
t2 = ('c', 'd')
vs.add(t1)
vs.add(t2)
q1 = 'bcdbbbb'
vs = sorted(vs, key=lambda k:
q1.__contains__(k[0]) * 100 + q1.__contains__(k[1]) * 10

            , reverse=True)

# for _ in vs:
# print(_)
