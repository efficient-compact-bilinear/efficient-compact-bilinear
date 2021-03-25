#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import torchvision

import cub200
from run import runLayer 
 
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class BCNN(torch.nn.Module):
    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=True).features#vgg16(pretrained=True).features
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])  # Remove pool5.

        self.d1 = 16#16#16#32
        self.d2 = 3#2#1#3

        self.w1 = torch.nn.Parameter((2 * torch.randint(0, 2, (self.d1, self.d2)) - 1).float())#, requires_grad = False)
        self.w2 = torch.nn.Parameter((2 * torch.randint(0, 2, (self.d1, self.d2)) - 1).float())#, requires_grad = False)
        self.w3 = torch.nn.Parameter((2 * torch.randint(0, 2, (self.d1, self.d2)) - 1).float())#, requires_grad = False)
        self.w4 = torch.nn.Parameter((2 * torch.randint(0, 2, (self.d1, self.d2)) - 1).float())
        self.fc = torch.nn.Linear((512//self.d1*self.d2)**2, 200)


        self.norm = runLayer()
        for param in self.features.parameters():
            param.requires_grad = False
        torch.nn.init.kaiming_normal(self.fc.weight.data)
        torch.nn.init.kaiming_normal(self.w1.data)
        torch.nn.init.kaiming_normal(self.w2.data)
        torch.nn.init.kaiming_normal(self.w3.data)
        torch.nn.init.kaiming_normal(self.w4.data)
 

        if self.fc.bias is not None:
            torch.nn.init.constant(self.fc.bias.data, val=0)
    def forward(self, X):
        eps = 1e-4
        N = X.size()[0]
        X = self.features(X)
        X = X.view(N, 512, 28**2)
        X = self.norm(X)
        X = X.permute(0,2,1).contiguous().view(-1,512)
        eps = 1e-5
        Xfold1 = X.view(-1,512//self.d1,self.d1)
        Xfold2 = Xfold1.permute([0,2,1]).contiguous().view(-1,512//self.d1,self.d1)#.permute([0,2,1])

        X1 = Xfold1.matmul(self.w1).view(N,-1,512//self.d1*self.d2)
        X2 = Xfold1.matmul(self.w2).view(N,-1,512//self.d1*self.d2)
        X3 = Xfold2.matmul(self.w3).view(N,-1,512//self.d1*self.d2)
        X4 = Xfold2.matmul(self.w4).view(N,-1,512//self.d1*self.d2)

        Xorg1 = X1.permute(0,2,1).bmm(X3).view(N,-1)
        Xorg2 = X2.permute(0,2,1).bmm(X4).view(N,-1)

        Xorg = Xorg1+Xorg2
        Xorg = torch.sqrt(Xorg.abs() + 1e-5).mul(Xorg.sign())
        X = torch.nn.functional.normalize(Xorg)
        X = self.fc(X)  
        return X


class BCNNManager(object):
    def __init__(self, options, path):
        print('Prepare the network and data.')
        self._options = options
        self._path = path
        # Network.
        self._net = torch.nn.DataParallel(BCNN()).cuda()
        print(self._net)
        # Criterion.
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Solver.
        self._solver = torch.optim.SGD(
            [self._net.module.w1, self._net.module.w2, self._net.module.w3, self._net.module.w4] + 
list(self._net.module.fc.parameters()), lr=self._options['base_lr'],
            momentum=0.9, weight_decay=self._options['weight_decay'])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._solver, mode='max', factor=0.1, patience=5, verbose=True,
            threshold=1e-4)

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),  # Let smaller edge match
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        train_data = cub200.CUB200(
            root=self._path['cub200'], train=True, download=True,
            transform=train_transforms)
        test_data = cub200.CUB200(
            root=self._path['cub200'], train=False, download=True,
            transform=test_transforms)
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=16, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=16,
            shuffle=False, num_workers=16, pin_memory=True)

    def train(self):
        best_acc = 0.0
        best_epoch = None
        for t in range(self._options['epochs']):
            epoch_loss = []
            num_correct = 0
            num_total = 0
            for X, y in self._train_loader:
                # Data.
                X = torch.autograd.Variable(X.cuda())
                y = torch.autograd.Variable(y.cuda())

                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
                loss = self._criterion(score, y)
                epoch_loss.append(loss.item())
                # Prediction.
                _, prediction = torch.max(score.data, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data)
                # Backward pass.
                loss.backward()
                self._solver.step()
            train_acc = 100 * num_correct / num_total
            test_acc = self._accuracy(self._test_loader)
            self._scheduler.step(test_acc)
            if test_acc > best_acc or t==80:
                best_acc = test_acc
                best_epoch = t + 1
                print('*', end='')
                # Save model onto disk.
                if t==80:
                    torch.save(self._net.state_dict(),
                           os.path.join(self._path['model'],
                                        'vgg_16_epoch_%d.pth' % (t + 1)))
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
                  (t+1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
        print('Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))

    def _accuracy(self, data_loader):
        """Compute the train/test accuracy.

        Args:
            data_loader: Train/Test DataLoader.

        Returns:
            Train/Test accuracy in percentage.
        """
        self._net.train(False)
        num_correct = 0
        num_total = 0
        for X, y in data_loader:
            # Data.
            X = torch.autograd.Variable(X.cuda())
            y = torch.autograd.Variable(y.cuda())

            # Prediction.
            score = self._net(X)
            _, prediction = torch.max(score.data, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.data).item()
        self._net.train(True)  # Set the model to training phase
        return 100 * num_correct / num_total

    def getStat(self):
        """Get the mean and std value for a certain dataset."""
        print('Compute mean and variance for training data.')
        train_data = cub200.CUB200(
            root=self._path['cub200'], train=True,
            transform=torchvision.transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=False, num_workers=4,
            pin_memory=True)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for X, _ in train_loader:
            for d in range(3):
                mean[d] += X[:, d, :, :].mean()
                std[d] += X[:, d, :, :].std()
        mean.div_(len(train_data))
        std.div_(len(train_data))
        print(mean)
        print(std)


def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train bilinear CNN on CUB200.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, required=True,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        required=True, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        required=True, help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        required=True, help='Weight decay.')
    args = parser.parse_args()
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
    }

    project_root = os.popen('pwd').read().strip()
    path = {
        'cub200': os.path.join(project_root, 'data/cub200'),
        'model': os.path.join(project_root, 'model_cub'),
    }
    for d in path:
        print(path[d])
        #assert os.path.isdir(path[d])

    manager = BCNNManager(options, path)
    # manager.getStat()
    manager.train()


if __name__ == '__main__':
    main()
