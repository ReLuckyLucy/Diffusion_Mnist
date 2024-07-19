import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# 设置matplotlib以正常显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录（如果不存在）
os.makedirs('output', exist_ok=True)

# 定义噪声函数
def corrupt(x, noise_amount):
    """按照给定的噪声量添加噪声到输入张量 x"""
    noise = torch.randn_like(x) * noise_amount.view(-1, 1, 1, 1)
    return x + noise

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'使用设备: {device}')

# 加载数据集
dataset = torchvision.datasets.MNIST(root="data/", train=True, download=True, transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
x, y = next(iter(train_dataloader))
print('输入形状:', x.shape)
print('标签:', y)
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')
plt.savefig('output/input_data.png')
plt.show()

# 定义 BasicUNet 类
class BasicUNet(nn.Module):
    """最小化的 UNet 实现"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([ 
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2), 
        ])
        self.act = nn.SiLU() # 激活函数
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x)) # 通过层和激活函数
            if i < 2: # 除了第三（最后一层）下采样层外的所有层
                h.append(x) # 存储输出用于跳跃连接
                x = self.downscale(x) # 下采样准备下一层
            
        for i, l in enumerate(self.up_layers):
            if i > 0: # 除了第一层上采样层外的所有层
                x = self.upscale(x) # 上采样
                x += h.pop() # 获取存储的输出（跳跃连接）
            x = self.act(l(x)) # 通过层和激活函数
            
        return x
    
# 初始化网络
net = BasicUNet()
x = torch.rand(8, 1, 28, 28)
print('输出形状:', net(x).shape)

# 检查网络中的参数数量
print('参数数量:', sum(p.numel() for p in net.parameters()))

# 数据加载器设置
batch_size = 128
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练设置
n_epochs = 3
net = BasicUNet()
net.to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
losses = []

# 训练循环
for epoch in range(n_epochs):
    for x, y in train_dataloader:
        x = x.to(device)
        noise_amount = torch.rand(x.shape[0]).to(device)
        noisy_x = corrupt(x, noise_amount)
        pred = net(noisy_x)
        loss = loss_fn(pred, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
    print(f'完成第 {epoch} 个周期。该周期的平均损失: {avg_loss:.5f}')

# 绘制并保存损失曲线
plt.plot(losses)
plt.ylim(0, 0.1)
plt.savefig('output/loss_curve.png')
plt.show()

# 获取一些数据
x, y = next(iter(train_dataloader))
x = x[:8]

# 按范围添加噪声
amount = torch.linspace(0, 1, x.shape[0])
noised_x = corrupt(x, amount)

# 获取模型预测
with torch.no_grad():
    preds = net(noised_x.to(device)).detach().cpu()

# 保存整个模型
save_dir = 'output/model'
os.makedirs(save_dir, exist_ok=True)

torch.save(net.state_dict(), 'output/model/basic_unet_model.pth')
print("模型已保存到 'output/model/basic_unet_model.pth'")

# 或者保存整个模型（包括优化器状态）
# torch.save({
#     'model_state_dict': net.state_dict(),
#     'optimizer_state_dict': opt.state_dict(),
#     'losses': losses,
#     'epoch': epoch
# }, 'output/basic_unet_model_full.pth')
# print("完整模型已保存到 'output/basic_unet_model_full.pth'")



# 绘制并保存结果
fig, axs = plt.subplots(3, 1, figsize=(12, 7))
axs[0].set_title('输入数据')
axs[0].imshow(torchvision.utils.make_grid(x)[0].clip(0, 1), cmap='Greys')
axs[1].set_title('添加噪声的数据')
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].clip(0, 1), cmap='Greys')
axs[2].set_title('网络预测')
axs[2].imshow(torchvision.utils.make_grid(preds)[0].clip(0, 1), cmap='Greys')
plt.savefig('output/predictions.png')
plt.show()
