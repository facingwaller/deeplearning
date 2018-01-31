import numpy as np
import pickle


class TextConverter(object):
    _type = ""

    def __init__(self, text=None, max_vocab=50000, filename=None, type=""):
        self._type = type

        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)
            # print(len(vocab))
            # max_vocab_process
            vocab_count = {}
            for word in vocab:
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1
            vocab_count_list = []
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    # 将test_list 转换成 int_array
    # text必须是字符串的list
    def text_to_arr_list(self, text):
        #
        if self._type == "zh-cn":
            # new_text = []
            # for w in text:
            #     new_text.append(w)
            text = [x for x in text[0]]
        if type(text) == str:
            text = str(text).strip()  # 去掉头尾空格
            text = str(text).split(" ")
        if type(text) != list:
            print("bad type")
            raise Exception("bad type not list %s" % type(list))
        #
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return arr

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def arr_to_text_by_space(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return " ".join(words)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)

    def save_to_file_raw(self, filename):
        with open(filename, 'w', encoding="utf-8") as f:
            for w in self.vocab:
                f.write(w + "\n")
