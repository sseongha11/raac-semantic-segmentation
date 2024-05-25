import torch.nn as nn
from transformers import ViTModel


class TransUNet(nn.Module):
    def __init__(self, n_classes):
        super(TransUNet, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Conv2d(48, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.vit(x)['last_hidden_state']  # Shape: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = x.size()

        # Remove the [CLS] token if present
        if seq_len == 197:
            x = x[:, 1:, :]  # Remove the first token (CLS token)
            seq_len -= 1

        height = width = int(seq_len ** 0.5)  # Should be 14 for 224x224 input

        if height * width != seq_len:
            raise ValueError(f"Expected seq_len to be a perfect square, got {seq_len} which is not.")

        # Reshape to [batch_size, hidden_size, height, width]
        x = x.permute(0, 2, 1).contiguous()  # Shape: [batch_size, hidden_size, seq_len]
        x = x.view(batch_size, hidden_size, height, width)  # Shape: [batch_size, 768, 14, 14]

        x = self.decoder(x)
        x = self.classifier(x)
        return x