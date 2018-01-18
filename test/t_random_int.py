import random


a = [2808,1]

index = 1
while True:
    b = random.randint(0,len(a)-1)
    print(a[b])
    index += 1
    if index == 111:
        break