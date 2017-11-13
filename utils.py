import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import scipy.misc as m

mean = np.array([103.939, 116.779, 123.68])


def cross_entropy2d(input, target, weight=None, size_average=True, ignore_index=-100):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False, ignore_index=ignore_index)
    if size_average:
        loss /= mask.data.sum()
    return loss


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu) # mean class IOU
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu,}, cls_iu


def get_tensor(img):
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= mean
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1)
    img_size = img.shape
    img = torch.from_numpy(img).float().view(1, img_size[0], img_size[1], img_size[2])
    return img


def decode_segmap(temp, plot=False):
        Sky = [128, 128, 128]
        Building = [128, 0, 0]
        Pole = [192, 192, 128]
        Road = [255, 69, 0]
        Pavement = [128, 64, 128]
        Tree = [60, 40, 222]
        SignSymbol = [128, 128, 0]
        Fence = [192, 128, 128]
        Car = [64, 64, 128]
        Pedestrian = [64, 0, 128]
        Bicyclist = [64, 64, 0]
        Unlabeled = [0, 128, 192]

        label_colours = np.array([Sky, Building, Pole, Road,
                                  Pavement, Tree, SignSymbol, Fence, Car,
                                  Pedestrian, Bicyclist, Unlabeled]).astype(np.uint8)
        r = np.zeros_like(temp).astype(np.uint8)
        g = np.zeros_like(temp).astype(np.uint8)
        b = np.zeros_like(temp).astype(np.uint8)
        for l in range(0, 12):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3)).astype(np.uint8)
        rgb[:, :, 0] = b
        rgb[:, :, 1] = g
        rgb[:, :, 2] = r
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb


def get_data_path(name):
    js = open('config.json').read()
    data = json.loads(js)
    return data[name]['data_path']