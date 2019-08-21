from tqdm import tqdm
import pandas as pd
import numpy as np
import config_m2det as C
import matplotlib.pyplot as plt
from collections import Counter
import os

df_train = pd.read_csv(C.TRAIN_CSV).dropna().reset_index()
unicode_map = {codepoint: char for codepoint, char in pd.read_csv(C.MAP_CSV).values}


def save_chars():
    chars = {}
    for i in range(df_train.shape[0]):
        try:
            a = [x for x in df_train.labels.values[i].split(' ') if x.startswith('U')]
            n_a = int(len(a))
            for j in a:
                if j not in chars:
                    chars[j] = 1
                else:
                    chars[j] += 1

            a = " ".join(a)

        except AttributeError:
            a = None
            n_a = 0

        df_train.loc[i, 'chars'] = a
        df_train.loc[i, 'n_chars'] = n_a

    # print(df_train.head())
    chars = pd.DataFrame(list(chars.items()), columns=['char', 'count'])
    chars['jp_char'] = chars['char'].map(unicode_map)
    # print(" >> Chars dataframe <<")
    # print("Number of chars: ", chars.shape[0])
    chars = chars.sort_values(by=['count'], ascending=False).reset_index(drop=True)
    chars.to_csv(C.CHARS_FREQ_FILE, index=True)
    # print(chars.head(20))


def chars_exist(func):
    def exist(*args):
        if not os.path.exists(C.CHARS_FREQ_FILE):
            save_chars()
        func(*args)

    return exist


@chars_exist
def find_the_balance(max_num=10000):
    df_chars = pd.read_csv(C.CHARS_FREQ_FILE)
    # df_chars = df_chars[df_chars['count'] > 10]
    # df_chars[df_chars['count'] < 50] = 50
    # df_chars[df_chars['count'] > max_num] = max_num
    total = df_chars['count'].sum()
    print("total :", total)
    sum = 0
    for i, row in df_chars.iterrows():
        count = row['count']
        sum += count
        if total - sum < 30000:
            print(i, row['char'])
            print(count)
            break
        # if max_num >= total - sum:
        #


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
    find_the_balance(1000)
