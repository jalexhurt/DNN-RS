import pretrainedmodels
import torch
from pretrainedmodels.models.inceptionresnetv2 import InceptionResNetV2
from pretrainedmodels.models.xception import Xception
from torchvision import models as pytorchmodels
from torchvision.models.resnet import ResNet

############
# Load the correct model
# and save the square edge
############
modelMap = {
    "ResNet18": pytorchmodels.resnet18,
    "ResNet50": pytorchmodels.resnet50,
    "ResNet101": pytorchmodels.resnet101,
    "Xception": pretrainedmodels.xception,
    "InceptionResNetV2": pretrainedmodels.inceptionresnetv2,
}


def get_model(model_name, num_classes, pretrained=True):
    """
    Get the model based upon name
     and then fix the model with num classes
    :param model_name: model name
    :param num_classes: number of classes
    :return: model ready for training
    """

    # get the model
    pretrained_model = modelMap[model_name]

    # params for a pretrained network
    pretrained_params = {"pretrained": True}
    # params for custom code
    init_params = {
        "Xception": {"pretrained": "imagenet"},
        "InceptionResNetV2": {"pretrained": "imagenet"},
    }

    # get params for this network
    params = init_params.get(model_name, pretrained_params)
    if "pretrained" in params and pretrained is False:
        params["pretrained"] = False
    # instantiate model
    model = pretrained_model(**params)

    # adjust classes in final FC layer
    if isinstance(model, ResNet):
        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif isinstance(model, Xception):
        model.last_linear = torch.nn.Linear(2048, num_classes)
    elif isinstance(model, InceptionResNetV2):
        model.avgpool_1a = torch.nn.AdaptiveAvgPool2d(1)
        model.last_linear = torch.nn.Linear(1536, num_classes)
    # return
    return model


def load_model(path, network, num_classes):
    """
    Load the model given the path and the name of the network
    :param gpus: which gpus to use
    :param num_classes: the number of classes
    :param path: path to the saved network
    :param network: the network name
    :return: the model, optimizer, epoch and loss
    """
    # load the model
    model = get_model(network, num_classes)
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    return model
