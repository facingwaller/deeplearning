

# 区号答案是 三位则需要加一位0
# 版本、价格等如果是  X.0 则可能  有X 和X.0两个答案 都需要抽取出来。
# 香港上市代码是 5位。如果是4位可能是缺第一个的0
#  + X
# A年B月C日

import re
a = '+ 1.00'
b = '01.00'
c = '1234年56月78日'

a = a.lower().replace(' ','')
# value = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
# result = value.match(a)

r_xm = re.findall('年([0-9])+月',c)
print(r_xm)

r1 = re.findall('^([0-9]){1,4}年([0-9]){1,2}月([0-9]){1,2}日$',c)

print(r1)

# if result:
#     print('float')
#     print(int(float(a)))
# else:
#     print('not float')

print(a.isdigit())
print(a.isnumeric())

