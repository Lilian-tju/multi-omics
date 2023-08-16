# Function: define the model of Autoencoder and DomainClassifier

import torch
import torch.nn as nn
import torch.nn.init as init

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.ReLU(),
            nn.Linear(300, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 300),
            nn.ReLU(),
            nn.Linear(300, input_dim),
        )
        #   HE权重初始化(kaiming_normal)
        for m in self.modules( ):
            if isinstance(m,nn.Linear):
                init.kaiming_normal(m.weight, nonlinearity='relu')


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# 域分类器模型
class DomainClassifier(nn.Module):
    def __init__(self, input_dim):
        super(DomainClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
#定义梯度反转层 
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            output = - alpha*grad_output
        return output, None
revgrad = GradientReversalLayer.apply