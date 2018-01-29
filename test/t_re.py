import re

p1 = r"((.*))"
p2 = '《[^《]*》'
p3 = '(\([^\((]*\))'
# re.findall('《[^《]*》',l1)

l1 = r"全国人增加《中华法》附件" \
     r"三所列全定(第十会议)122321312312 " \
     r"《1》 （3） (1)"

# r1 = re.search(p1,l1)


print("------------")
r2 = re.findall(p3, l1)
# print(r2)
for r in r2:
    print(r)
