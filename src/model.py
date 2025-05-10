# Modèle 3D basique utilisé dans train_glasses.py
from torch import nn

class Simple3DModel(nn.Module):
    def __init__(self):
        super(Simple3DModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(128 * 64 * 64, 3 * 2048)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, 2048, 3)
        return x
