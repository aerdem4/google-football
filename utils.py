import math
import numpy as np


def length(vec):
    return math.sqrt(np.sum(vec*vec))


def distance(source, target):
    return length(np.array(target) - np.array(source))


def angle(source, target):
    diff = np.array(target) - np.array(source)
    return 360*np.arctan2(diff[1], diff[0])/(2*np.pi) % 360


def cosine_sim(v1, v2):
    return (v1*v2).sum()/length(v1)/length(v2)


def between(x, a, b, threshold=-0.9):
    x = np.array(x)
    return cosine_sim(np.array(a) - x, np.array(b) - x) < threshold
