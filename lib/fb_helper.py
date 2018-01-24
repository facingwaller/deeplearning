# coding=utf-8
import codecs
import logging
import gzip
import json
import numpy as np
import os
# 抽取fb_5m里面的实体

class freebase:
    # 列出FB_2M 中所有提及的 entity 和 relation 输出到 data/fb2m
    # www.freebase.com/m/018fj69	www.freebase.com/music/recording/artist	www.freebase.com/m/01wbgdv
    @staticmethod
    def extract_fbxm(file_name='../data/fb2m/freebase-FB2M.txt'):
        f1_writer = codecs.open("../data/fb2m/e.txt", mode="w", encoding="utf-8")
        f2_writer = codecs.open("../data/fb2m/r.txt", mode="w", encoding="utf-8")
        e1 = []
        r1 = []
        index = 0
        with codecs.open(file_name, mode="r", encoding="utf-8") as read_file:
            for line in read_file.readlines():
                index += 1
                if index % 10000 == 0:
                    print("==============", index)
                count = 0
                try:

                    count = len(line.split('\t'))
                except Exception as e2:
                    print(line)
                    print(index)
                    print(e2)
                    return

                if count != 3:
                    print(count)
                    continue
                line = line.replace("www.freebase.com/", "").replace("\n", "")
                # f1_writer.write(line.split('\t')[0])
                e1.append(line.split('\t')[0])
                e2 = line.split('\t')[2]
                e2_s = e2.split(' ')
                if len(e2_s) > 0:
                    for e2_s1 in e2_s:
                        e1.append(e2_s1)
                else:
                    e1.append(e2)
                r1.append(line.split('\t')[1])
                # print("==========")

        for _ in e1:
            f1_writer.write(_ + "\n")
        for _ in r1:
            f2_writer.write(_ + "\n")

        f1_writer.close()
        f2_writer.close()

    # -----从fb_raw里面抽取出entity_name作为subject的所有rdf
    @staticmethod
    def extract_rdf_from_fb(entity_name):
        print(1)
if __name__ == "__main__":
    freebase.excat_fbxm()