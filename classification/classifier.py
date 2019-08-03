from model import VGG_ResNet
import config as C
import torch
from torchvision.transforms import ToTensor, Compose


class Classifier:
    def __init__(self, checkpoint):
        self.classifier = VGG_ResNet(C.NUM_CLASSES).cuda()
        self.classifier.load_state_dict(
            torch.load(checkpoint)
        )

    def predict(self, imgs):
        imgs = torch.Tensor(imgs)/255
        imgs = imgs.cuda()
        self.classifier.eval()
        output = self.classifier.forward(imgs)
        label = torch.argmax(output, 1)
        return label.cpu().detach().numpy()

