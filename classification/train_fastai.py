from fastai import *
from fastai.vision import *
from dataloader import dataloader
from model import VGG_ResNet
import config as C


os.environ["CUDA_VISIBLE_DEVICES"] = '3'

num_classes = 100
default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = VGG_ResNet
data = DataBunch(dataloader(True), dataloader(False), device=default_device)
# data.show_batch()
learn = ConvLearner(data, model, pretrained=False, metrics=accuracy)

# lr_find(learn)
fit_one_cycle(learn,100, 0.001)
