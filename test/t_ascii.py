a = 'citrus medic  l.	香橼(中药)	拉丁学名	Citrus medic L.'
print(ascii(a))

s = '&nbsp;&nbsp;T-shirt\xa0\xa0短袖圆领衫,体恤衫\xa0'
out = "".join(s.split())
print(out)

s=s.replace('&nbsp;','')
print(s)

