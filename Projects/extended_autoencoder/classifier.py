import torch.nn as nn

class ClassifierModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, output_size=4, dropout_rate=0.5):
        """
        A feedforward classifier that takes a 128-dimensional latent vector (from the VAE encoder)
        and outputs logits for 4 classes.
        Hidden_size is increased to 256 to provide more capacity.
        """
        super(ClassifierModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # x should be the latent representation extracted from the VAE encoder.
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)  # Raw logits
        return x
