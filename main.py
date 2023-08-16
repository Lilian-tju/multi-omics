#使用numpy读取数据，主要是确定三个域信息
import pandas as pd
import torch
import numpy as np
from model import Autoencoder, DomainClassifier, GradientReversalLayer
import torch.optim as optim


# 读取数据
# 读取三个CSV文件
domain1_data = pd.read_csv('D:\python\multi_omics\data\降维\encoded_exp.csv')
domain2_data = pd.read_csv('D:\python\multi_omics\data\降维\encoded_methy.csv')
domain3_data = pd.read_csv('D:\python\multi_omics\data\降维\encoded_cna.csv')

# 将数据转换为NumPy数组
domain1_array = domain1_data.to_numpy()
domain2_array = domain2_data.to_numpy()
domain3_array = domain3_data.to_numpy()

domain_data = []
domain_data.append(domain1_array)
domain_data.append(domain2_array)
domain_data.append(domain3_array)

num_domains = len(domain_data)
num_samples = domain_data[0].shape[0]
input_dim = domain_data[0].shape[1]
bottleneck_dim = 64
num_epochs = 100
learning_rate = 0.001

# 定义域标签
labels = []
labels_domain1 = np.zeros(domain1_array.shape[0])  # 域1的标签为0
labels_domain2 = np.ones(domain2_array.shape[0])  # 域2的标签为1
labels_domain3 = np.ones(domain3_array.shape[0]) * 2  # 域3的标签为2
labels.append(labels_domain1)
labels.append(labels_domain2)
labels.append(labels_domain3)

# for domain in range(num_domains):
#     labels.extend([domain] * num_samples)等价于上边的代码，保证可读性，任选其一即可

data_tensor = torch.tensor(np.concatenate(domain_data), dtype=torch.float32)

#标准化
mean = data_tensor.mean()
std = data_tensor.std()
data_tensor = (data_tensor - mean) / std

# 使用unsqueeze(1)函数将标签的维度从(batch_size,77)增加到(batch_size, 1)，使其与域分类器模型的输出形状相匹配。
# 通过这个修改，标签的形状应该与模型的输出形状匹配
labels_tensor = torch.tensor(np.concatenate(labels), dtype=torch.float32).unsqueeze(1)

# 定义模型和优化器
autoencoder = Autoencoder(input_dim, bottleneck_dim)
domain_classifiers = DomainClassifier(bottleneck_dim)
ae_criterion = nn.MSELoss()
dc_criterion = nn.BCELoss()
ae_optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
dc_optimizers = optim.Adam(domain_classifiers.parameters(), lr=learning_rate)

# 训练模型
# torch.autograd.set_detect_anomaly(True)

for epoch in range(num_epochs):
    autoencoder.train()
    domain_classifiers.train()

    # 训练自编码器
    ae_optimizer.zero_grad()
    reconstructed, encoded = autoencoder(data_tensor)
    ae_loss = ae_criterion(reconstructed, data_tensor)
    ae_loss.backward(retain_graph=True)#
    ae_optimizer.step()

    # 训练域分类器
    dc_optimizers.zero_grad()
    reversed_encoded = GradientReversalLayer.apply(encoded.detach().clone(), 1.0)
    domain_preds = domain_classifiers(reversed_encoded.detach().clone())
    dc_loss = dc_criterion(domain_preds, labels_tensor)
    dc_loss.backward()
    dc_optimizers.step()

    # 输出训练过程信息
   # if (epoch + 1) % 10 == 0:
    print(f"Epoch [{epoch+1}/{num_epochs}], Autoencoder Loss: {ae_loss.item():.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Domain Classifier Loss: {dc_loss.item():.4f}")

# 示例用法：提取特征
autoencoder.eval()
with torch.no_grad():
    _, encoded = autoencoder(data_tensor)
    features = encoded.numpy()

print("提取的特征：")
print(features)
