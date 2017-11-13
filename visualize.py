import cv2
import sys
import numpy as np
import argparse
from model.LinkNet import LinkNet
import torch
from torch.autograd import Variable
from datasets.loader import get_loader
from utils import get_tensor, decode_segmap
import time


def processImage(infile, args):
    n_classes = 12
    model = LinkNet(n_classes)
    model.load_state_dict(torch.load(args.model_path))
    if torch.cuda.is_available():
        model = model.cuda(0)
    model.eval()
    gif = cv2.VideoCapture(infile)
    cv2.namedWindow('camvid')
    while (gif.isOpened()):
        ret, frame = gif.read()
        frame = cv2.resize(frame, (768, 576))
        images = get_tensor(frame)
        if torch.cuda.is_available():
            images = Variable(images.cuda(0))
        else:
            images = Variable(images)
        outputs = model(images)
        pred = outputs.data.max(1)[1].cpu().numpy().reshape(576, 768)
        pred = decode_segmap(pred)
        vis = np.zeros((576, 1536, 3), np.uint8)
        vis[:576, :768, :3] = frame
        vis[:576, 768:1536, :3] = pred
        cv2.imshow('camvid', vis)
        cv2.waitKey(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='linknet_camvid_39.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='camvid', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    args = parser.parse_args()
    processImage('datasets/camvid/seq2_fast.gif', args)