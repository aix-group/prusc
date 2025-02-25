import torch
from munch import Munch
from model.simple_model import CNN, MLP, FC
from model.resnet import ResNet18, ResNet34
from model.wide_resnet import WideResNet28_10, WideResNet16_8

from prune.GateSimpleModel import GateCNN, GateFCN
from prune.GateResnet import GateResNet18, GateResNet34, LowPassGateResNet18, GateResNet50
from prune.GateWideResnet import GateWideResNet28_10, GateWideResNet16_8

from data.transforms import num_classes

def build_model(args):
    n_classes = num_classes[args.data]
    
    if args.data == 'cmnist':
        classifier = GateCNN() if not args.cmnist_use_mlp else GateFCN()
        biased_classifier = GateCNN() if not args.cmnist_use_mlp else GateFCN()
        nets = Munch(classifier=classifier,
                         biased_classifier=biased_classifier)
    else:
        classifier = GateResNet50(IMAGENET_pretrained=args.imagenet, n_classes=n_classes)
        biased_classifier = GateResNet50(IMAGENET_pretrained=args.imagenet, n_classes=n_classes)
        nets = Munch(classifier=classifier,
                         biased_classifier=biased_classifier)
    return nets
