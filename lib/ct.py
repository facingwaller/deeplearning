import random


class ct:
    @staticmethod
    def random_get_one_from_list(list):
        return list[random.randint(0, len(list) - 1)]

    @staticmethod
    def test_random_get_one_from_list():
        l = [1, 2, 3]
        print(ct.random_get_one_from_list(l))

    @staticmethod
    def max_of_line(question_list):
        return max([len(x.split(" ")) for x in question_list])

    # ---------------去除不对的格式
    @staticmethod
    def replace_noise(lines):
        return [str(l).replace("/", "_").replace("_", " ") for l in lines]
    
    #------------格式化单行的关系格式
    @staticmethod
    def clear_relation(relation):
        return relation.replace("/", "_").replace("_", " ").strip()

    #----------------用指定数字填充或者截断
    @staticmethod
    def padding_line(ling,max_len,padding_num):
        padding = max_len - len(s)
        for index in range(padding):
            s.append(padding_num)  


if __name__ == "__main__":
    ct.test_random_get_one_from_list()
