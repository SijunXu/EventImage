from __future__ import print_function
import time
import copy
import torch

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    since = time.time()
    history = {'loss':[], 'val_loss':[], 'acc':[], 'val_acc':[]}
    model = model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {0}/{1}'.format(epoch, num_epochs - 1))
        #### training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        for i, (inputs, target) in enumerate(dataloaders['train']):
            inputs = inputs.to(device).float()
            target = target.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum((outputs+.5).int().t() == target.data.int().t())
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_corrects = running_corrects.double() / len(dataloaders['train'].dataset)
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_corrects)
        
        #### validation phase
        val_running_loss = 0.0
        val_running_corrects = 0.0
        model.eval()
        for i, (inputs, target) in enumerate(dataloaders['val']):
            inputs = inputs.to(device).float()
            target = target.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum((outputs+.5).int().t() == target.data.int().t())
        val_epoch_loss = val_running_loss / len(dataloaders['val'].dataset)
        val_epoch_corrects = val_running_corrects.double() / len(dataloaders['val'].dataset)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_corrects)

        print('Epoch Loss: {0:.6f}, Acc: {1:.6f}, Val Loss: {2:.6f}, Val Acc: {3:.6f}'.
              format(epoch_loss, epoch_corrects, val_epoch_loss, val_epoch_corrects))
        print('-' * 10)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, history

def train_J2_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    since = time.time()
    history = {'loss':[], 'val_loss':[], 'acc':[], 'val_acc':[]}
    model = model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {0}/{1}'.format(epoch, num_epochs - 1))
        #### training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        for i, (inputs, fw, target) in enumerate(dataloaders['train']):
            inputs = inputs.to(device).float()
            fw = fw.to(device).float()
            target = target.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs, fw)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum((outputs+.5).int().t() == target.data.int().t())
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_corrects = running_corrects.double() / len(dataloaders['train'].dataset)
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_corrects)
        
        #### validation phase
        val_running_loss = 0.0
        val_running_corrects = 0.0
        model.eval()
        for i, (inputs, fw, target) in enumerate(dataloaders['val']):
            inputs = inputs.to(device).float()
            fw = fw.to(device).float()
            target = target.to(device).float()
            outputs = model(inputs, fw)
            loss = criterion(outputs, target)
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum((outputs+.5).int().t() == target.data.int().t())
        val_epoch_loss = val_running_loss / len(dataloaders['val'].dataset)
        val_epoch_corrects = val_running_corrects.double() / len(dataloaders['val'].dataset)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_corrects)

        print('Epoch Loss: {0:.6f}, Acc: {1:.6f}, Val Loss: {2:.6f}, Val Acc: {3:.6f}'.
              format(epoch_loss, epoch_corrects, val_epoch_loss, val_epoch_corrects))
        print('-' * 10)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, history
