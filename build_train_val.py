import pandas as pd
import numpy as np
import config as C

np.random.seed(C.SEED)

df_train = pd.read_csv(C.TRAIN_CSV).dropna()
image_ids = df_train['image_id'].values
length = len(image_ids)
idx = np.arange(length)
np.random.shuffle(idx)
SPILT = int(0.7 * length)

with open(C.TRAIN_FILE, 'w') as train_file:
    with open(C.VAL_FILE, 'w') as val_file:
        train_file.write('\n'.join(image_ids[idx[:SPILT]]))
        val_file.write('\n'.join(image_ids[idx[SPILT:]]))
