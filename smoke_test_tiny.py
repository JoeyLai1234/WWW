import argparse
import torch
import torchvision as tv
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./datasets/tiny-imagenet-200/train', help='Path to Tiny ImageNet train folder')
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    transform = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = tv.datasets.ImageFolder(args.data_root, transform=transform)
    print(f'Found {len(dataset)} images across {len(dataset.classes)} classes')

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # load model from repo's models.resnet (keeps compatibility with the project weights API)
    from models.resnet import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            out = model(images)
            print('Input batch shape:', images.shape)
            print('Output shape:', out.shape)
            # print top-1 preds and a few examples
            probs = torch.nn.functional.softmax(out, dim=1)
            top1 = probs.argmax(dim=1)
            print('Top-1 predictions (indices):', top1[:8].cpu().numpy())
            break

    print('Smoke test completed.')


if __name__ == '__main__':
    main()
