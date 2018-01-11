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


if __name__ == "__main__":
    ct.test_random_get_one_from_list()
