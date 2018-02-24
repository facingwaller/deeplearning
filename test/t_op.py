

with open("1",mode='w',encoding='utf-8') as f1:
    f1.write("123\n")
    f1.write("123\n")
    f1.write("123\n")

with open("1",mode='r',encoding='utf-8') as f1:
    for l in f1.readlines():
        print(l)
