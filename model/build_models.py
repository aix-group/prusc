import torch
from munch import Munch
from model.resnet import ResNet18, ResNet34

from prune.GateResnet import GateResNet18, GateResNet34, LowPassGateResNet18, GateResNet50
from prune.GateWideResnet import GateWideResNet28_10, GateWideResNet16_8

from data.transforms import num_classes

def build_model(args):
    n_classes = num_classes[args.data]
    classifier = GateResNet50(IMAGENET_pretrained=args.imagenet, n_classes=n_classes)
    biased_classifier = GateResNet50(IMAGENET_pretrained=args.imagenet, n_classes=n_classes)
    nets = Munch(classifier=classifier, biased_classifier=biased_classifier)
    return nets
