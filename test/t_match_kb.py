# 区号答案是 三位则需要加一位0
# 版本、价格等如果是  X.0 则可能  有X 和X.0两个答案 都需要抽取出来。
# 香港上市代码是 5位。如果是4位可能是缺第一个的0
#  + X
# A年B月C日

import time, datetime
import re
from lib.ct import ct

a = '+ 1.00'
b = '01.00'
b2 = '01的 2001年2月1日 '
c = "2000年06月08日"
d = '0001年20月20日'
f = '2000年6月8日'

a = a.lower().replace(' ', '')
for item in [a, b, b2, ' 2.1100 ', '张3']:
    print(ct.padding_int(item))
print(ct.padding_date(b2))
# t = time.strptime("2009 - 08 - 08", "%Y - %m - %d")
# y ,m ,d = t[0:3]
# print(datetime.datetime(y ,m ,d))
#
# print(ct.padding_date(f))
# print(ct.padding_date(c))
# print(ct.padding_date(d))

# r_xm = re.findall('年([0-9])+月' ,c)
# print(r_xm)
#
# r1 = re.findall('^([0-9]){1,4}年([0-9]){1,2}月([0-9]){1,2}日$' ,c)
#
#
# print(r1)


# print(a.isdigit())
# print(a.isnumeric())
