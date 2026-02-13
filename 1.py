import numpy as np
import pandas as pd
import torch

alpha=-1
beta=1
delta=0.3
gamma=0.37
omega=1.2


# 1---变换（将数据集转换为tensor 类型）---
df=pd.read_csv('duffing_samples.csv')
#print(df.head())
np_data=df.to_numpy()
data=torch.tensor(np_data,dtype=torch.float32)
#print(data)

# 2---构建神经网络---
# 获取训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义神经网络
from torch import nn
#import torch.nn.functional as F
class NeuralNetwork(nn.Module):    # 定义一个继承自 nn.Module 的神经网络类。nn.Module 是 PyTorch 中所有神经网络模块的基类
    def __init__(self):
        super(NeuralNetwork, self).__init__()    # 构造函数，初始化父类（nn.Module）的所有功能。
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2,256),   # 输入层
            nn.ReLU(),    # 激活函数，引入非线性
            nn.Linear(256, 256),    # 隐藏层
            nn.ReLU(),    # 激活函数，引入非线性
            nn.Linear(256, 3),    # 输出层
        )

    def forward(self, x):
        l = self.linear_relu_stack(x)  # 输出向量 [B, 3]
        l11 = torch.log(1 + torch.exp(l[:, 0]))   # 保证正值
        l21 = l[:, 1]
        l22 = torch.log(1 + torch.exp(l[:, 2]))
        L = torch.zeros(x.size(0), 2, 2, device=x.device)
        L[:, 0, 0] = l11
        L[:, 1, 0] = l21
        L[:, 1, 1] = l22

        eps = 1e-3
        P = torch.bmm(L.transpose(1, 2), L) + eps * torch.eye(2).to(x.device).unsqueeze(0)
        return P

# 实例化模型并放到设备上
model = NeuralNetwork()


import torch
import torch.nn.functional as F

def observer_loss(model, x, k=10):
    """
    x: [B, 2]     输入状态点
    model:        输出 P(x) 的神经网络
    k:            固定观测器增益（默认 1.0）
    """

    x = x.clone().detach().requires_grad_(True)  # 保证可求导
    P = model(x)                                 # 输出对称正定矩阵 P(x), shape: [B, 2, 2]

    # --- 计算 df/dx ---
    def f(x):
        x1 = x[:, 0:1]  # 保持 shape [B, 1]
        x2 = x[:, 1:2]
        dx1 = x2
        dx2 = -delta * x2 - alpha * x1 - beta * x1 ** 3
        return torch.cat([dx1, dx2], dim=1)  # shape [B, 2]

    fx = f(x)
    B = x.shape[0]
    jacobians = []


    # 计算f(x)的雅可比矩阵dfx/dx
    for i in range(fx.shape[1]):  # 对每个输出维度
        grad_outputs = torch.zeros_like(fx)
        grad_outputs[:, i] = 1.0
        grads = torch.autograd.grad(fx, x, grad_outputs=grad_outputs,
                                    retain_graph=True, create_graph=True)[0]
        jacobians.append(grads.unsqueeze(2))

    J = torch.cat(jacobians, dim=2)  # [B, 2, 2]

    # --- 计算 dh/dx ---
    hx = x[:, 0:1]  # h(x) = x1
    dh = torch.autograd.grad(hx, x, grad_outputs=torch.ones_like(hx),
                             retain_graph=True, create_graph=True)[0]  # [B, 2]
    dh_dx = dh.unsqueeze(1)  # [B, 1, 2]

    # --- 计算 dP/dx ---
    P_trace = torch.diagonal(P, dim1=1, dim2=2).sum(dim=1)  # trace(P)
    dP_trace_dx = torch.autograd.grad(P_trace, x, grad_outputs=torch.ones_like(P_trace),
                                      retain_graph=True, create_graph=True)[0]  # [B, 2]
    dP_dx = dP_trace_dx.view(B, 1, 2).repeat(1, 2, 1)  # 近似构造 dP/dx [B, 2, 2]

    # --- 构造 M(x) ---
    PTJ = torch.bmm(P, J.transpose(1, 2))  # [B, 2, 2]
    PJ = torch.bmm(P, J)
    HTH = torch.bmm(dh_dx.transpose(1, 2), dh_dx)  # [B, 2, 2]
    M = dP_dx + PJ + PTJ - k * HTH  # [B, 2, 2]

    # --- 最大特征值作为损失 ---
    eigvals = torch.linalg.eigvalsh(M)  # [B, 2]，升序
    lam_max = eigvals[:, -1]            # 最大特征值
    loss_per_sample = F.relu(lam_max)   # ReLU 处理（>0 才算违反负定）

    # --- 平均损失 ---
    loss = loss_per_sample.mean()
    return loss,lam_max


# 3.---训练---
# 把数据集分为训练集和验证集
total_samples=len(data)
training_zie=int(total_samples * 0.8)  # The first 80% is the training set

train_data=data[0:training_zie]
test_data=data[training_zie:]

# 开始训练
# 构建DataLoader
from torch.utils.data import DataLoader, TensorDataset

train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data), batch_size=32)
# 配置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# 编写训练循环
EPOCHS = 100

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        x_batch = batch[0].to(device)  # shape: [B, 2]
        x_batch.requires_grad_(True)
        loss,lam_max = observer_loss(model, x_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    # 验证
    model.eval()
    val_loss = 0.0
    for batch in test_loader:
        x_val = batch[0].to(device)
        x_val.requires_grad_(True)
        loss,_ = observer_loss(model, x_val)
        val_loss += loss.item()
    avg_val_loss = val_loss / len(test_loader)


    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    print("requires_grad:", x_batch.requires_grad)

# 求K值
def find_min_k(model, data, k_start=0.0, k_end=100.0, k_step=0.1):
    model.eval()    # 切换到评估模式
    data = data.to(device)

    def compute_lambda_max(model, x, k):
        x = x.clone().detach().requires_grad_(True)
        P = model(x)

        # 计算 f(x)
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        dx1 = x2
        dx2 = -delta * x2 - alpha * x1 - beta * x1 ** 3
        fx = torch.cat([dx1, dx2], dim=1)

        # df/dx
        B = x.shape[0]
        jacobians = []
        for i in range(fx.shape[1]):
            grad_outputs = torch.zeros_like(fx)
            grad_outputs[:, i] = 1.0
            grads = torch.autograd.grad(fx, x, grad_outputs=grad_outputs,
                                        retain_graph=True, create_graph=True)[0]
            jacobians.append(grads.unsqueeze(2))
        J = torch.cat(jacobians, dim=2)

        # dh/dx
        hx = x[:, 0:1]
        dh = torch.autograd.grad(hx, x, grad_outputs=torch.ones_like(hx),
                                 retain_graph=True, create_graph=True)[0]
        dh_dx = dh.unsqueeze(1)

        # dP/dx
        P_trace = torch.diagonal(P, dim1=1, dim2=2).sum(dim=1)
        dP_trace_dx = torch.autograd.grad(P_trace, x, grad_outputs=torch.ones_like(P_trace),
                                          retain_graph=True, create_graph=True)[0]
        dP_dx = dP_trace_dx.view(B, 1, 2).repeat(1, 2, 1)

        # M(x)
        PTJ = torch.bmm(P, J.transpose(1, 2))
        PJ = torch.bmm(P, J)
        HTH = torch.bmm(dh_dx.transpose(1, 2), dh_dx)
        M = dP_dx + PJ + PTJ - k * HTH

        eigvals = torch.linalg.eigvalsh(M)  # shape: [B, 2]
        lam_max = eigvals[:, -1]
        return lam_max

    for k in np.arange(k_start, k_end, k_step):
        lam_max = compute_lambda_max(model, data, k)
        if torch.all(lam_max < 0):
            print(f"Found minimal k = {k:.4f} where all λ_max(M(x)) < 0")
            return k
    print("No suitable k found in given range.")
    return None


print('The Final value of K: ', find_min_k(model, test_data, k_start=0.0, k_end=100.0, k_step=0.1))



import scipy.io as sio
# 假设 data 是你测试用的 x_grid 数据 [N, 2]
# 并且 model 是训练好的 PyTorch 网络

model.eval()
x_grid = data.numpy()
with torch.no_grad():
    P_tensor = model(data)  # shape: [N, 2, 2]
    P_np = P_tensor.numpy()

sio.savemat("P_interp_data.mat", {
    'x_grid': x_grid,  # shape: [N, 2]
    'P': P_np          # shape: [N, 2, 2]
})






