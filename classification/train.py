import datetime

from dataloader import dataloader
from model import VGG_ResNet
from pytorchtools import EarlyStopping
import numpy as np
import config as C
import os
import torch
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

writer = SummaryWriter(os.path.join(C.LOG_ROOT, f'{C.NUM_CLASSES}_{datetime.datetime.now().strftime("%m-%d %H-%M-%S")}'))
default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = VGG_ResNet(num_classes=C.NUM_CLASSES).cuda()
train_dataloader = dataloader(True)
val_dataloader = dataloader(False)
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []

optimizer = torch.optim.Adam(model.parameters())
early_stopping = EarlyStopping(patience=20, verbose=True)
tbs = 0
vbs = 0
# model.load_state_dict(torch.load('checkpoint.pt'))
print("开始训练")
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(C.EPOCHES):
    model.train()
    t_cor = 0
    t_total = 0
    for batch, (data, target) in enumerate(train_dataloader, 1):
        data = data.cuda()
        target = target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = loss_fn(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # record training loss
        train_losses.append(loss.item())
        t_cor += np.sum((torch.argmax(output, 1) == target).cpu().detach().numpy())
        t_total += len(output.cpu().detach().numpy())
        writer.add_scalar('Batch' + '/' + 'train_loss', loss.item(), tbs)
        tbs += 1
        print(f'第 {epoch} epoch, batch {batch}  loss: {loss.item()}')
    model.eval()  # prep model for evaluation
    cor = 0
    total = 0
    for data, target in val_dataloader:
        data = data.cuda()
        target = target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = loss_fn(output, target)
        # record validation loss
        valid_losses.append(loss.item())
        cor += np.sum((torch.argmax(output, 1) == target).cpu().detach().numpy())
        total += len(output.cpu().detach().numpy())
        writer.add_scalar('Batch' + '/' + 'val_loss', loss.item(), vbs)
        vbs += 1
    val_acc = cor/total
    train_acc = t_cor/t_total
    print(f'epoch  {epoch}  train_acc   {train_acc}  val_acc  {val_acc}')
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    log_info = [
        ('train_loss', train_loss),
        ('val_loss', valid_loss),
        ('train_acc', train_acc),
        ('val_acc', val_acc)
    ]

    for k, v in log_info:
        print(v)
        writer.add_scalar('Epoch' + '/' + k, v, epoch)
    early_stopping(valid_loss, model, f'{C.NUM_CLASSES}_{epoch}')
    if early_stopping.early_stop:
        print("Early stopping")
        break

torch.save(model.state_dict(), 'best_checkpoint.pt')