#coding=utf-8
from xml.etree import ElementTree

#打开xml文档
# dom = xml.dom.minidom.parse('data/fudan_questions.xml')
import codecs
# text = open("data/fudan_questions1.xml").read().decode("utf-8")
# text=open('data/fudan_questions.xml').read()

f1 = open('data/fudan_questions1.xml','r', encoding='UTF-8')
text = f1.read()
root = ElementTree.fromstring(text)
#得到文档元素对象

# 1 通过getiterator
lst_node = root.getiterator("DOC")
for node in lst_node:
    for n2 in node.getchildren():
        print(n2.text)
# itemlist = root.getElementsByTagName('DOC')
# for item in itemlist:
#     # print(item.nodeName)
#     for child in item.getchildren():
#         print(child.data)

