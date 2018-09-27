import re

p1 = r"((.*))"
p2 = '《[^《]*》'
p3 = '(\([^\((]*\))'
# re.findall('《[^《]*》',l1)

l1 = r"全国人增加《中华法》附件" \
     r"三所列全定(第十会议)122321312312 " \
     r"《1》 （3） (1)"
l2 = '(啊|呀(你知道)？吗|呢)?(?|\?)*$'
# r1 = re.search(p1,l1)

print(len('♠(xeone5-2609/4gb/500gb/quadro600)的显示内存是多少?'))
print("------------")
r2 = re.findall(p3, l1)
# print(r2)
for r in r2:
    print(r)
