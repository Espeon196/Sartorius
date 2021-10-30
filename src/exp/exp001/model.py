import timm
from torch import nn
import torch


class BrainTumor2dModel(nn.Module):
    def __init__(self, model_arch, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features=n_features, out_features=1, bias=True)

    def forward(self, x):
        out = self.model(x)
        return out

if __name__ == "__main__":
    model = BrainTumor2dModel(model_arch='efficientnet_b0', pretrained=False)
    batchsize, C, H, W = 2, 3, 256, 256
    input = torch.randn(batchsize, C, H, W)
    logit = model(input)
    print(logit.shape)
    from torchsummary import summary
    print(summary(model, (3, 256, 256)))