

def t1():
    t2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                t3 = ("%d_%d_%d"%(i,j,k))
                t2.append(t3)
                print(t3)
                if len(t2)%3 == 0 and len(t2)!=0:
                    t4 = t2.copy()
                    t2.clear()
                    yield t4
gc1 = t1()
for gc in gc1:
    print(gc)