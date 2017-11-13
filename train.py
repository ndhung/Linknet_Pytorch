import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from model.LinkNet import LinkNet
from utils import cross_entropy2d, scores, get_data_path
from datasets.loader import get_loader
from torch.optim.lr_scheduler import StepLR

#hist = np.array([508907015, 83944159, 315210837, 9042293, 12102630, 16945509, 2870627,
#                 7613617, 220046942, 15972632, 55628186, 16824978, 1865213, 96511472,
#                 3691448, 3246703, 3214814, 1361413, 5711755, 179044557]).astype('float')
hist = np.array([10682767, 14750079, 623349, 20076880, 2845085, 6166762, 743859, 714595, 3719877, 405385, 184967, 2503995]).astype('float')


def train(args):
    data_path = get_data_path(args.dataset)
    data_loader = get_loader(args.dataset)
    loader = data_loader(data_path, is_transform=True)
    n_classes = loader.n_classes
    trainloader = data.DataLoader(
        loader, batch_size=args.batch_size)
    another_loader = data_loader(data_path, split='test', is_transform=True)
    valloader = data.DataLoader(
        another_loader, batch_size=args.batch_size)
    
    # compute weight for cross_entropy2d
    norm_hist = hist / np.max(hist)
    weight = 1 / np.log(norm_hist + 1.02)
    weight[-1] = 0
    weight = torch.FloatTensor(weight)
    model = LinkNet(n_classes)
    
    if torch.cuda.is_available():
        model.cuda(0)
        weight = weight.cuda(0)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr_rate)
    scheduler = StepLR(optimizer, step_size=100, gamma=args.lr_decay)
    for epoch in xrange(args.epochs):
        scheduler.step()
        model.train()
        loss_list = []
        file = open('logs' + '/{}_{}.txt'.format('linknet', epoch), 'w')
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(0))
                labels = Variable(labels.cuda(0))
            else:
                images = Variable(images)
                labels = Variable(labels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = cross_entropy2d(outputs, labels, weight=weight)
            loss_list.append(loss.data[0])
            loss.backward()
            optimizer.step()
        
        # file.write(str(np.average(loss_list)))
        print(np.average(loss_list))
        file.write(str(np.average(loss_list)) + '\n')
        model.eval()
        gts, preds = [], []
        for i, (images, labels) in enumerate(valloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(0))
                labels = Variable(labels.cuda(0))
            else:
                images = Variable(images)
                labels = Variable(labels)
            outputs = model(images)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.data.cpu().numpy()
            for gt_, pred_ in zip(gt, pred):
                gts.append(gt_)
                preds.append(pred_)
        score, class_iou = scores(gts, preds, n_class=n_classes)
        for k, v in score.items():
            file.write('{} {}\n'.format(k, v))

        for i in range(n_classes):
            file.write('{} {}\n'.format(i, class_iou[i]))
        torch.save(model.state_dict(), "logs" + "/{}_{}_{}.pkl".format(
            'linknet', args.dataset, epoch))
        file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperprams')
    parser.add_argument('--epochs', nargs='?', type=int, default=300,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=8,
                        help='Batch Size')
    parser.add_argument('--lr_rate', nargs='?', type=float, default=5e-4,
                        help='Learning Rate')
    parser.add_argument('--w_decay', nargs='?', type=float, default=2e-4,
                        help='Weight Decay')
    parser.add_argument('--momentum', nargs='?', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--lr_decay', nargs='?', type=float, default=1e-1,
                        help='Learning Rate Decay')                    
    parser.add_argument('--dataset', nargs='?', type=str, default='camvid', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    args = parser.parse_args()
    train(args)
