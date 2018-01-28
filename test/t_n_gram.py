# import urllib2
import re
import string
import operator

def cleanText(input):
    input = re.sub('\n+', " ", input).lower() # 匹配换行,用空格替换换行符
    input = re.sub('\[[0-9]*\]', "", input) # 剔除类似[1]这样的引用标记
    input = re.sub(' +', " ", input) #  把连续多个空格替换成一个空格
    # input = bytes(input).encode('utf-8') # 把内容转换成utf-8格式以消除转义字符
    #input = input.decode("ascii", "ignore")
    return input

def cleanInput(input):
    input = cleanText(input)
    cleanInput = []
    input = input.split(' ') #以空格为分隔符，返回列表


    for item in input:
        item = item.strip(string.punctuation) # string.punctuation获取所有标点符号

        if len(item) > 1 or (item.lower() == 'a' or item.lower() == 'i'): #找出单词，包括i,a等单个单词
            cleanInput.append(item)
    return cleanInput

def getNgrams(input, n):
    # input = cleanInput(input)

    output = {} # 构造字典
    for i in range(len(input)-n+1):
        ngramTemp = " ".join(input[i:i+n])#.encode('utf-8')
        if ngramTemp not in output: #词频统计
            output[ngramTemp] = 0 #典型的字典操作
        output[ngramTemp] += 1
    return output
content = "把符合的记录的实体于问题中的字符串匹配看是否存在于问题中，从问题库中用答案匹配知识库中的记录"
#content = open("1.txt").read()
ngrams = getNgrams(content, 2)
print(ngrams)


ngrams = getNgrams(content, 3)
print(ngrams)

# sortedNGrams = sorted(ngrams.items(), key = operator.itemgetter(1), reverse=True) #=True 降序排列
# print(sortedNGrams)