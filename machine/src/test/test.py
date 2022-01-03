import random

x = [k for k in range(20)]


def aiueo():
    for i in range(30):
        yield i, i ** 2


for i in aiueo():
    print(i)