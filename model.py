"""CNN model definition for the 16-class scene recognition assignment."""

import torch.nn as nn


def conv_bn_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TNet(nn.Module):
    """Four stages of conv-BN-ReLU (x2) with max pooling, then GAP + linear head."""

    def __init__(self, num_classes=16, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        c1, c2, c3, c4 = widths

        self.features = nn.Sequential(
            # Stage 1
            conv_bn_relu(3, c1),
            conv_bn_relu(c1, c1),
            nn.MaxPool2d(2),
            # Stage 2
            conv_bn_relu(c1, c2),
            conv_bn_relu(c2, c2),
            nn.MaxPool2d(2),
            # Stage 3
            conv_bn_relu(c2, c3),
            conv_bn_relu(c3, c3),
            nn.MaxPool2d(2),
            # Stage 4 (no pool; global average pooling follows)
            conv_bn_relu(c3, c4),
            conv_bn_relu(c4, c4),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = TNet(num_classes=16)
    print(net)
    print(f"Trainable parameters: {count_parameters(net):,}")
