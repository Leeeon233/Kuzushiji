import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
import config as C
from PIL import Image

np.random.seed(C.SEED)


class ClassificationDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.image_names = []
        self.labels = []
        self.df_train_val = pd.read_csv(C.CHARS_DICT_FILE)
        self.df_train_val.columns = ['img_name', 'label', 'jp_char', 'is_train']
        self.transform = transform
        # self.target_transform = target_transform
        self._init_data(train)

    def _init_data(self, train):
        print('初始化数据')
        if train:
            df_train = self.df_train_val[self.df_train_val['is_train'] == 1]
            num = C.NUM_PER_CLASS
        else:
            df_train = self.df_train_val[self.df_train_val['is_train'] == 0]
            num = C.NUM_PER_CLASS // 20
        for i in range(C.NUM_CLASSES - 1):
            need = df_train[df_train['label'] == (i + 1)][['img_name', 'label']].values
            np.random.shuffle(need)
            need = need[:num]
            self.image_names += list(need[:, 0])
            self.labels += list(need[:, 1])
            # self.image_names += names
        others = df_train[df_train['label'] > C.NUM_CLASSES - 1][['img_name', 'label']].values
        np.random.shuffle(others)
        others = others[:num]
        self.image_names += list(others[:, 0])
        self.labels += [0] * num

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(C.SAVE_ALL_CHARS, f'{image_name}.jpg')
        label = self.labels[index]
        img = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)

    def __len__(self):
        return len(self.image_names)


def dataloader(train):
    batch_size = C.BATCH_SIZE if train else C.BATCH_SIZE // 8
    tfms = Compose([Resize((224, 224)), ToTensor()])
    return DataLoader(ClassificationDataset(train, tfms),
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=8)


if __name__ == '__main__':
    ClassificationDataset(True).__getitem__(1)
