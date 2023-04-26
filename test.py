import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import random
import matplotlib.pyplot as plt
from model.utils.util import get_img

DATA_DIR = 'D:\Coding-Workspace\MonteCosmoLocalization\model\data'


SAMPLE_SIZE = 72

for _ in range(500):
    id_A = 5
    id_P = (id_A + (1 if random.random() < 0.5 else -1)) % SAMPLE_SIZE
    id_N = random.choice(list(set(range(SAMPLE_SIZE)) - set([id_A-1, id_A, id_A+1])))
    print(id_N, end=' ')