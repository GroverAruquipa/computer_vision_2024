import torch
import torch.nn as nn

class cnn_fasteners_classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(43808, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
        )

    def forward(self, x, verbose=False):
        out = self.network(x)

        if verbose:
          print(out.shape)
        return out