# 另一台机器（10.3.x.x）完整操作指南

> 本文档为新设备提供从零开始的完整配置流程。
> 目标：让该机器能够同步代码、阅读研究进展、并在需要时参与训练。

---

## 第一步：基础软件安装

### 1.1 安装 Git
```powershell
# 下载并安装 Git for Windows
# https://git-scm.com/download/win

# 验证安装
git --version
```

### 1.2 安装 Miniconda（Python 管理）
```powershell
# 下载 Miniconda
# https://docs.conda.io/en/latest/miniconda.html

# 安装后打开 Anaconda Prompt，验证
conda --version
python --version
```

### 1.3 安装 VSCode
```powershell
# 下载 VSCode
# https://code.visualstudio.com/

# 安装推荐插件（在 VSCode 扩展商店搜索安装）：
# - Python (Microsoft)
# - Remote-SSH (Microsoft)
# - GitLens (GitKraken)
```

---

## 第二步：Python 环境配置

### 2.1 创建虚拟环境
```powershell
# 打开 Anaconda Prompt 或 PowerShell
conda create -n torch_cu121 python=3.10 -y
conda activate torch_cu121
```

### 2.2 安装 PyTorch（CUDA 12.1）
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 安装项目依赖
```powershell
pip install rasterio numpy tqdm pillow labelme
```

### 2.4 验证安装
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"
python -c "import rasterio; print('rasterio OK')"
python -c "import labelme; print('labelme OK')"
```

**预期输出：**
```
PyTorch: 2.5.1+cu121
CUDA可用: True
GPU: NVIDIA GeForce RTX xxxx
rasterio OK
labelme OK
```

---

## 第三步：Git 和 GitHub 配置

### 3.1 配置 Git 用户信息
```powershell
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱@example.com"
```

### 3.2 克隆项目仓库
```powershell
# 选择一个工作目录，例如 D:\projects 或 E:\projects
cd D:\projects  # 或你喜欢的路径

# 克隆仓库
git clone https://github.com/bboyWang/tp0630.git

# 进入项目目录
cd tp0630
```

### 3.3 验证同步
```powershell
# 查看远程仓库
git remote -v

# 拉取最新代码
git pull origin master

# 查看提交历史
git log --oneline -5
```

**预期输出：**
```
origin  https://github.com/bboyWang/tp0630.git (fetch)
origin  https://github.com/bboyWang/tp0630.git (push)
56ca498 v8: 完整项目同步 - 模型训练+主动学习代码+文档+研究日志+环境配置清单
75f585b 初始化项目
```

---

## 第四步：数据同步

数据文件较大（约数 GB），**不通过 GitHub 同步**，需要手动拷贝。

### 4.1 数据目录结构
```
E:\tp0630\                    # 或 D:\tp0630，根据你的盘符调整
├── tiff\                     # 1504 个 30 波段 TIFF 文件
├── eiseg_ready\images\       # PNG 预览图 + JSON 标注
├── checkpoints_v7\           # v7 模型权重
├── checkpoints_v8\           # v8 模型权重
├── ready.txt                 # 标注注册表
├── results_v8.txt            # 训练结果
├── uncertain_v7.txt          # v7 不确定性排名
└── uncertain_v8.txt          # v8 不确定性排名
```

### 4.2 数据传输方式（任选其一）

**方式 A：局域网共享（推荐）**
```powershell
# 在台式机（已拥有数据的机器）上：
# 1. 右键 E:\tp0630 -> 属性 -> 共享 -> 高级共享
# 2. 勾选"共享此文件夹"，设置共享名为 "tp0630"
# 3. 权限设置为"读取"（或"读取/写入"如果需要修改）

# 在另一台机器（10.3.x.x）上：
# 1. 打开文件资源管理器，地址栏输入：
\\台式机IP地址\tp0630

# 2. 右键映射网络驱动器，例如映射为 Z: 盘
# 3. 以后可以通过 Z:\ 访问数据
```

**方式 B：移动硬盘拷贝**
```powershell
# 直接将台式机的 E:\tp0630 文件夹完整复制到移动硬盘
# 再从移动硬盘复制到另一台机器的 E:\tp0630
```

**方式 C：SCP 传输（如果两台机器都开 SSH）**
```powershell
# 在另一台机器上执行（需要台式机开启 SSH 服务）
scp -r 用户名@台式机IP:E:\tp0630 E:\
```

### 4.3 验证数据
```powershell
# 检查数据是否完整
Get-ChildItem E:\tp0630\tiff | Measure-Object  # 应显示 1504 个文件
Get-ChildItem E:\tp0630\eiseg_ready\images\*.png | Measure-Object  # 应显示约 1504 个文件
```

---

## 第五步：项目代码配置

### 5.1 修改数据路径（如果盘符不同）
如果另一台机器的数据盘符不是 `E:`，需要修改代码中的路径：

**文件：`dataset.py` 第 13 行**
```python
# 原代码
C = {
    'data_dir': r'E:\tp0630',  # ← 如果数据在 D 盘，改为 r'D:\tp0630'
    ...
}
```

**文件：`train_active.py` 第 13 行**
```python
sys.path.insert(0, r'E:\tp0630')  # ← 同样修改盘符
```

### 5.2 创建本地配置文件（可选）
为了避免每次都修改代码，可以创建一个本地配置：

**创建 `config_local.py`（不提交到 GitHub）**
```python
# config_local.py
# 本地机器特定配置，已加入 .gitignore

DATA_DIR = r'E:\tp0630'  # 根据实际数据位置修改
SAVE_DIR = r'E:\tp0630\checkpoints_v9'
```

**修改 `train_active.py` 开头**
```python
import os, sys

# 尝试导入本地配置，如果不存在则使用默认配置
try:
    from config_local import DATA_DIR, SAVE_DIR
    print(f"[Config] 使用本地配置: {DATA_DIR}")
except ImportError:
    DATA_DIR = r'E:\tp0630'
    SAVE_DIR = r'E:\tp0630\checkpoints_v8'
    print(f"[Config] 使用默认配置: {DATA_DIR}")

# ... 后续代码
```

**更新 `.gitignore`**
```bash
# 在 .gitignore 末尾添加
config_local.py
```

---

## 第六步：验证项目运行

### 6.1 测试数据加载
```powershell
conda activate torch_cu121
cd D:\projects\tp0630  # 你的项目路径

python -c "
from dataset import WetlandDataset
ds = WetlandDataset(r'E:\tp0630', patch_size=256, quadrant_mode=False)
print(f'数据集样本数: {len(ds)}')
print(f'人类标注样本: {sum(1 for s in ds.samples if s[\"is_human\"])}')
"
```

**预期输出：**
```
数据集样本数: 1504
人类标注样本: 196
```

### 6.2 测试模型加载
```powershell
python -c "
from model import UNet
import torch
model = UNet(in_channels=26, num_classes=3, base_ch=32)
print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')
print(f'模型设备: {\"cuda\" if torch.cuda.is_available() else \"cpu\"}')"
```

**预期输出：**
```
模型参数量: 3,307,907
模型设备: cuda
```

### 6.3 测试完整训练流程（可选）
```powershell
# 注意：训练需要 GPU 和完整数据，首次运行可能需要较长时间
python train_active.py
```

---

## 第七步：阅读研究进展

### 7.1 查看最新进展
```powershell
# 拉取最新代码
git pull origin master

# 阅读研究日志
notepad PROGRESS.md
# 或在 VSCode 中打开
code PROGRESS.md
```

### 7.2 关键文档阅读顺序
```
1. PROGRESS.md          ← 最先读，了解最新进展和下一步
2. project_knowledge.txt ← 项目完整知识库
3. SETUP.md             ← 环境配置参考
4. v8_summary.txt       ← 当前轮次详细结果
```

---

## 第八步：日常工作流程

### 8.1 每日开始工作前
```powershell
cd D:\projects\tp0630
git pull origin master  # 同步最新代码和文档
```

### 8.2 每日工作结束后
```powershell
# 如果有代码修改
git add -A
git commit -m "描述你的修改"
git push origin master

# 如果只是阅读，无需提交
```

### 8.3 更新研究日志（重要）
每次完成重要工作后，更新 `PROGRESS.md`：
```powershell
# 编辑 PROGRESS.md，添加新的日期条目
# 记录：做了什么、结果如何、下一步计划

git add PROGRESS.md
git commit -m "docs: 更新研究进展 - 简要描述"
git push origin master
```

---

## 第九步：远程服务器连接（如需要）

如果另一台机器需要通过 VSCode 远程连接同局域网的服务器：

### 9.1 安装 Remote-SSH 插件
- 在 VSCode 扩展商店搜索 "Remote-SSH" 并安装

### 9.2 配置 SSH 连接
```powershell
# 编辑 SSH 配置文件
notepad C:\Users\你的用户名\.ssh\config
```

**添加以下内容：**
```
Host myserver
    HostName 10.3.x.x  # 服务器实际 IP
    User 你的用户名
    Port 22
```

### 9.3 连接服务器
- 在 VSCode 中按 `F1`
- 输入 "Remote-SSH: Connect to Host"
- 选择 "myserver"

---

## 第十步：故障排查

### 问题 1：git push 失败（权限）
```powershell
# 方案 A：使用 GitHub Personal Access Token
# 1. 访问 https://github.com/settings/tokens
# 2. 生成新 token（勾选 repo 权限）
# 3. 推送时用 token 代替密码

# 方案 B：配置 SSH 密钥
ssh-keygen -t ed25519 -C "你的邮箱"
# 将公钥添加到 GitHub: https://github.com/settings/keys
```

### 问题 2：CUDA 不可用
```powershell
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 如果版本不匹配，重新安装对应版本的 PyTorch
```

### 问题 3：rasterio 安装失败
```powershell
# 使用 conda 安装
conda install -c conda-forge rasterio
```

### 问题 4：数据路径错误
```powershell
# 检查数据是否存在
Test-Path E:\tp0630\tiff
Test-Path E:\tp0630\eiseg_ready\images

# 如果不存在，检查数据是否已正确拷贝
```

---

## 快速检查清单

配置完成后，运行以下命令确认一切就绪：

```powershell
# 1. Git 同步
cd D:\projects\tp0630
git pull origin master

# 2. Python 环境
conda activate torch_cu121

# 3. 验证代码
python -c "from dataset import WetlandDataset; print('dataset OK')"
python -c "from model import UNet; print('model OK')"

# 4. 验证数据
python -c "import os; print(f'TIFF: {len(os.listdir(r\"E:\tp0630\tiff\"))} files')"

# 5. 阅读进展
notepad PROGRESS.md
```

全部通过后，环境配置完成！

---

*最后更新: 2026-07-30*
*适用机器: 10.3.x.x（Windows，VSCode 远程控制服务器）*
