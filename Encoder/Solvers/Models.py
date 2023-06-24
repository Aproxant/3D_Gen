import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, dict_size):
        super(TextEncoder, self).__init__()

        self.embedding = nn.Embedding(dict_size, 128)
        self.conv_128 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.bn_128 = nn.BatchNorm2d(128)
        self.conv_256 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1, bias=False),
            nn.ReLU()
        )
        self.bn_256 = nn.BatchNorm2d(256)

        self.lstm = nn.LSTM(256, 256, batch_first=True)

        self.outputs = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, inputs):

        embedded = self.embedding(inputs) # (batch_size, seq_size, emb_size)

        conved = self.conv_128(embedded.transpose(2, 1).contiguous()) # (batch_size, emb_size, seq_size)

        conved = self.bn_128(conved.unsqueeze(3)).squeeze() # (batch_size, emb_size, seq_size)
        conved = self.conv_256(conved) # (batch_size, emb_size, seq_size)

        conved = self.bn_256(conved.unsqueeze(3)).squeeze().transpose(2, 1).contiguous() # (batch_size, seq_size, emb_size)

        encoded, _ = self.lstm(conved, None)

        outputs = self.outputs(encoded[:, -1, :].squeeze())

        return outputs

class ShapeEncoder(nn.Module):
    def __init__(self):
        super(ShapeEncoder, self).__init__()
        self.network = nn.Sequential(
                nn.Conv3d(4, 64, 3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm3d(64),
                nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm3d(128),
                nn.Conv3d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm3d(256)
            )
        self.outputs = nn.Linear(256, 128)


    def forward(self, inputs):
        output = self.network(inputs)

        pooled = output.view(inputs.size(0), output.size(1), -1).contiguous().mean(2)

        outputs = self.outputs(pooled.view(pooled.size(0), -1))

        return outputs
