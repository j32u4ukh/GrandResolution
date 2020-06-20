import numpy as np
import tensorflow as tf
from tensorflow.math import (
    add,
    subtract,
    multiply,
    divide,
    square,
    pow as tf_pow,
    reduce_mean as tf_mean,
    reduce_std as tf_std
)


def multiOperation(op, *args):
    result = args[0]
    length = len(args)

    for i in range(1, length):
        result = op(result, args[i])

    return result


def log(x, base=None):
    if base is None:
        return tf.math.log(x)
    else:
        return divide(tf.math.log(x), tf.math.log(base))


# 最大公因數
def gcd(a, b):
    # https://www.geeksforgeeks.org/gcd-in-python/
    while b > 0:
        a, b = b, a % b

    return a


# 最小公倍數
def lcm(a, b):
    # http://drweb.nksh.tp.edu.tw/student/lesson/G005/
    return int(a * b / gcd(a, b))


if __name__ == "__main__":
    # 最大公因數
    print(gcd(a=20, b=30))

    # 最小公倍數
    print(lcm(a=6, b=9))

    log_e = log(np.e)
    log_10 = log(100., 10.)
    log_2 = log(8., 2.)
    with tf.Session() as sess:
        print("log_e:", sess.run(log_e))
        print("log_10:", sess.run(log_10))
        print("log_2:", sess.run(log_2))
