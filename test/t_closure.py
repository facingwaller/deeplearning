#!/usr/bin/env python
# coding=utf-8


def generate_counter():
    cnt = [0]

    def add_one():
        cnt[0] = cnt[0] + 1
        return cnt[0]

    return add_one


counter = generate_counter()
print(counter())  # 1
print(counter())
print(counter())
