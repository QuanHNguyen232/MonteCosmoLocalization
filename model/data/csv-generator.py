import os
import pandas as pd
import numpy as np
from collections import defaultdict

def generate_csv(datadir: str) -> None:
    ''' Generate csv file from datadir
    Args:
        datadir: directory that has data
    Return:
        None - metadata.csv is created under datadir
    '''
    data = defaultdict(list)
    for dir in os.listdir(DATA_DIR):
        if not os.path.isdir(os.path.join(DATA_DIR, dir)): continue
        img_id_list = [int(file.split('.')[0]) for file in os.listdir(os.path.join(DATA_DIR, dir)) if (file.endswith('jpg') and file[0].isdigit())]    # file = 'img_id.jpg'
        img_id_list = sorted(img_id_list)
        for img in img_id_list:
            data['sample_id'].append(dir)
            data['image_id'].append(img)
            data['sample_size'].append(len(img_id_list))
    df = pd.DataFrame(data=data)
    df.to_csv(os.path.join(datadir, 'metadata.csv'))

if __name__ == '__main__':
    DATA_DIR = './model/data'
    generate_csv(DATA_DIR)