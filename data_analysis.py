from tqdm import tqdm
import pandas as pd
import numpy as np
import config as C
import matplotlib.pyplot as plt
from collections import Counter

df_train = pd.read_csv(C.TRAIN_CSV).dropna()
unicode_map = {codepoint: char for codepoint, char in pd.read_csv(C.MAP_CSV).values}


def main():
    ws = np.empty((0, 1))
    hs = np.empty((0, 1))
    for idx in tqdm(range(len(df_train))):
        img_name, labels = df_train.values[idx]
        labels = np.array(labels.split(' ')).reshape(-1, 5)
        w = labels[:, 3].reshape((-1, 1))
        h = labels[:, 4].reshape((-1, 1))
        ws = np.concatenate([ws, w], axis=0)
        hs = np.concatenate([hs, h], axis=0)
    ws = ws.flatten().tolist()
    hs = hs.flatten().tolist()
    print(Counter(ws))
    print(Counter(hs))

if __name__ == '__main__':
    main()