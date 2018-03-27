my_alphabet = ['a', 'b', 'c']

def custom_key(word):
   numbers = []
   for letter in word:
      numbers.append(my_alphabet.index(letter))
   return numbers
# python中的整数列表能够比较大小
# custom_key('cbaba')==[2, 1, 0, 1, 0]

x=['cbaba', 'ababa', 'bbaa']
x.sort(key=custom_key)

x1 = ['a','bb','ccc']
def get_total(word):
    return len(word)
x1.sort(key=get_total,reverse=True)
print(x1)