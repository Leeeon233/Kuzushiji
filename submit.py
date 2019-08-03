import os
import pandas as pd
import config as C
from tqdm import tqdm

if not os.path.exists(C.SUBMIT_FILE):
    pd.DataFrame(columns=['image_id', 'labels']).to_csv(C.SUBMIT_FILE, mode='w', index=False)


def save2submit(image_id, label):
    save_content = {
        'image_id': image_id,
        'labels': label
    }
    char_dict = pd.DataFrame(save_content, index=[0])
    char_dict.to_csv(C.SUBMIT_FILE, mode='a', header=None, index=False)
    # with open(os.path.join(C.SUBMIT_ROOT, f'{image_id}.txt'), 'w') as f:
    #     f.write(label)


for file_name in tqdm(os.listdir(C.SUBMIT_ROOT)):
    image_id = file_name[:-4]
    with open(os.path.join(C.SUBMIT_ROOT, file_name), 'r') as f:
        label = f.readlines()
        if len(label) > 0:
            label = label[0].strip()
        else:
            label = ''
    save2submit(image_id, label)
