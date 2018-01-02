# coding=utf-8
import codecs
import logging
import gzip
import json
import numpy as np
import os


# 从文件中读取问题集合
# 返回句子和标签
def read_files(file_name):
    line_list = []
    idx = 0
    entity1_list = []
    relation_list = []
    entity2_list = []
    question_list = []

    with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
        try:
            for line in read_file.readlines():
                idx += 1
                line_seg = line.split('\t')
                # www.freebase.com/m/04whkz5
                entity1 = line_seg[0].split('/')[2]
                relation1 = line_seg[1].replace("www.freebase.com/", "")
                entity2 = line_seg[2].split('/')[2]
                question = line_seg[3].replace("\r\n", "")

                entity1_list.append(entity1)
                relation_list.append(relation1)
                entity2_list.append(entity2)
                question_list.append(question)
                # check it
                line_list.append(line)
        except Exception as e:
            print("index = ", idx)
            logging.error("error ", e)
    logging.info("load embedding finish!")
    return entity1_list, relation_list, entity2_list, question_list


class free_base:
    entitys = []

    def init_fb(self, file_name="../data/freebase/freebase_entity.txt"):
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                self.entitys.append(line.replace("\r\n",""))

    def find_fb_by_id(self,id):
        exist = False

        for e1 in self.entitys:

            if str(e1) == str(id):
                exist = True
                print(e1, id)
        return exist




def read_fb(entity_name):
    file_name = "../data/freebase/freebase_entity.txt"
    return


# 从文件中压缩包中读取指定的entity
def find_fb_by_id2(entity_name):
    print("-->entity_name")
    file_name = "../data/freebase/freebase_entity.txt"
    print(file_name)
    lines = []
    with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
        for line in read_file.readlines():
            lines.append(line)

    return


def test1():
    # read_files("../data/sq/annotated_fb_data_train-1.txt")
    # read_fb("1111.json")
    print("1"=="1")
    fb1 = free_base()
    fb1.init_fb()
    print(fb1.entitys.__len__())
    r = fb1.find_fb_by_id("012_0k9")
    print(r)

    return


if __name__ == "__main__":
    test1()
