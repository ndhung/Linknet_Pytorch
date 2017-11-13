import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time

from torch.autograd import Variable
from torch.utils import data
from datasets.loader import get_loader
from utils import get_data_path, scores
from model.LinkNet import LinkNet


def validate(args):

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True)
    n_classes = loader.n_classes
    valloader = data.DataLoader(loader, batch_size=args.batch_size)

    # Setup Model
    model = LinkNet(n_classes)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    if torch.cuda.is_available():
        model.cuda(0)

    gts, preds = [], []
    for i, (images, labels) in enumerate(valloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda(0))
            labels = Variable(labels.cuda(0))
        else:
            images = Variable(images)
            labels = Variable(labels)
        t1 = time.time()
        outputs = model(images)
        t2 = time.time()
        print(t2 - t1)
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()
        
        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)
    score, class_iou = scores(gts, preds, n_class=n_classes)
    for k, v in score.items():
        print k, v

    for i in range(n_classes):
        print i, class_iou[i] 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='linknet_camvid_39.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='camvid', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='test', 
                        help='Split of dataset to test on')
    args = parser.parse_args()
    validate(args)