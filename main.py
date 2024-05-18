#code based on here
#https://github.com/kuangliu/pytorch-cifar/tree/master
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random

from utils import progress_bar
import resnet18k
import mcnn


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', default=False, type=bool,
                    help='resume from checkpoint')
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch size')
parser.add_argument('--model', default="resnet18k", type=str,
                    help='specify model name')
parser.add_argument('--optimizer', default="adam", type=str,
                    help='specify optimizer')
parser.add_argument('--noise', default=0.0, type=float,
                    help='data noise ratio')
parser.add_argument('--data_size', default=1.0, type=float,
                    help='data size ratio')
parser.add_argument('--data', default="cifar10", type=str,
                    help='specify dataset name')
parser.add_argument('--w_param', default=64, type=int,
                    help='layer width parameter')

args = parser.parse_args()
epoch = 1
main_path = f'{args.model}_{args.data}_{args.optimizer}/noise-{args.noise}_datasize-{args.data_size}_w_param-{args.w_param}'
num_classes = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def prepare_dataset():
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.data == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        new_len = int(args.data_size*len(trainset))
        if args.data_size < 1.0:
            trainset, _ = torch.utils.data.random_split(trainset, [new_len, len(trainset)-new_len])
        if args.noise > 0.0:
            noise_len = int(len(trainset) * args.noise)
            index = torch.randperm(len(trainset))[:noise_len]
            for i in index:
                noise_label = (trainset.targets[i]-9) * (-1)
                trainset.targets[i] = noise_label

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
        
    elif args.data == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        
        new_len = int(args.data_size*len(trainset))
        if args.data_size < 1.0:
            trainset, _ = torch.utils.data.random_split(trainset, [new_len, len(trainset)-new_len])
        if args.noise > 0.0:
            noise_len = int(len(trainset) * args.noise)
            index = torch.randperm(len(trainset))[:noise_len]
            for i in index:
                noise_label = (trainset.targets[i]-9) * (-1)
                trainset.targets[i] = noise_label

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
        
    return trainloader, testloader

# Model
print('==> Building model..')
if args.model == "resnet18k":
    net = resnet18k.make_resnet18k(c = args.w_param, num_classes = num_classes)
    net = net.to(device)
elif args.model == "mcnn":
    net = mcnn.make_cnn(c = args.w_param, num_classes = num_classes)
    net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume==True:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(os.path.join(main_path, 'checkpoint')), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(main_path,'./checkpoint/ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
if args.optimizer == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=5e-4)
    epoch = 2000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
elif args.optimizer == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    epoch = 2000

train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

os.makedirs(main_path, exist_ok=True)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_accuracy = 100.*correct/total

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            test_accuracy = 100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.join(main_path,'checkpoint')):
            os.mkdir(os.path.join(main_path,'checkpoint'))
        torch.save(state, os.path.join(main_path,'./checkpoint/ckpt.pth'))
        best_acc = acc

        torch.save({
        'train_loss_history': train_loss_history,
        'train_accuracy_history': train_accuracy_history,
        'test_loss_history': test_loss_history,
        'test_accuracy_history': test_accuracy_history,
        }, os.path.join(main_path,'history.pth'))

if __name__ == "__main__":
    trainloader, testloader = prepare_dataset()
    for epoch in range(start_epoch, start_epoch+epoch):
        train(epoch)
        test(epoch)
        if args.optimizer=='sgd':
            scheduler.step()