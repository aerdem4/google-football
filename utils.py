import math
import numpy as np


def length(vec):
    return math.sqrt(np.sum(vec*vec))


def distance(source, target):
    return length(np.array(target) - np.array(source))


def angle(source, target):
    diff = np.array(target) - np.array(source)
    return 360*np.arctan2(diff[1], diff[0])/(2*np.pi) % 360
