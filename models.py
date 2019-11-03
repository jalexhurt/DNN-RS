import torch
from torchvision.models import resnet50


class Classifier20(torch.nn.Module):
    def forward(self, *input):
        raise NotImplementedError


def classifier20():
    return Classifier20()


_models = {
    "Classifier20": classifier20,
    "ResNet50": resnet50
}

models = list(_models.keys())


def Model(model_name, *model_args, **model_kwargs):
    if model_name not in models:
        raise ValueError("Model {} does not exist")
    ModelClass = _models[model_name]
    return ModelClass(*model_args, **model_kwargs)
