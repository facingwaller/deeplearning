from lib.baike_helper import baike_helper, baike_test
from lib.classification_helper import classification
from lib.data_helper import DataClass
from lib.config import config
from lib.ct import ct





class pretreatment:

    @staticmethod
    def re_write(f1,f2):
        """将问题格式转换"""
        f1s = ct.file_read_all_lines_strip(f1)
        f2s = []
        for l1 in f1s:
            if str(l1).__contains__('question id'):
                f2s.append(str(l1).split('\t')[1].replace(' ','').lower())
        ct.file_wirte_list(f2,f2s)

    @staticmethod
    def stat_all_space(f1):
        f1s = ct.file_read_all_lines_strip(f1)
        t1 = 0
        for l1 in f1s:
            l1_len = len(str(l1).split('\t'))
            if l1_len ==1:
                t1+=1
        print(t1)
        pass








