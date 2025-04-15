import os
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    def __init__(self, latent_dims, num_epochs):
        super(Decoder, self).__init__()

        self.model_file = os.path.join(
            f"model/epochs_{str(num_epochs)}_latent_{str(latent_dims)}/",
            "decoder_model.pth",
        )
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)

        # Fully connected layers to expand the latent vector
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dims, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 9 * 4 * 512),
            nn.LeakyReLU(),
        )

        # Unflatten to (512, 4, 9) — double check that (4, 9) is correct
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 4, 9))

        # Transposed convolutions to upsample back to 80x160
        # We reverse the channel order: 512 -> 256 -> 128 -> 64 -> 3
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1, output_padding=(0, 1)
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, output_padding=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1, output_padding=(0, 0)
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                64, 3, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x

    def save(self):
        torch.save(self.state_dict(), self.model_file)
        print(f"Decoder model saved to {self.model_file}")

    def load(self):
        self.load_state_dict(torch.load(self.model_file, map_location=device))
        print(f"Decoder model loaded from {self.model_file}")
