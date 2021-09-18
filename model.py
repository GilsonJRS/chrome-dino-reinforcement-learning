from torch import nn 
import copy

class DinoNet(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        channels, height, width = input_dim

        self.qnet = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136,512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.qhatnet = copy.deepcopy(self.qnet)
        for p in self.qhatnet.parameters():
            p.requires_grad = False
        
    def forward(self, input, model):
        if model == 'qnet':
            return self.qnet(input)
        elif model == 'qhatnet':
            return self.qhatnet(input)