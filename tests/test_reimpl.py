import torch

from reimpl.simple_resnet import simple_resnet18


def test_simple_resnet_forward():
    model = simple_resnet18(num_classes=100)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 100), f"unexpected output shape: {out.shape}"
