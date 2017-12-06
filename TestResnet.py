from model.Bilinear_Res import Bilinear_Res
from utils import cross_entropy2d, scores, get_data_path
from datasets.loader import get_loader
import os.path as osp
from torchvision import models
from utils import cross_entropy2d, scores, get_data_path, get_log_dir
here = osp.dirname(osp.abspath(__file__))

model = models.resnet101(True)
for name, layer in model.named_children():
    print(name, layer)