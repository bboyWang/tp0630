# 环境配置清单

> 本文档列出本项目所需的所有软件和 Python 库，供新设备（10.3.x.x）配置参考。
> 请按顺序安装，并验证每一步。

---

## 一、系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 |
| GPU | NVIDIA GPU（建议 4GB+ 显存，RTX 3050 即可） |
| Python | 3.9+（推荐 3.10） |
| Git | 最新版 |

---

## 二、必需软件

### 1. Git
- 下载：https://git-scm.com/download/win
- 验证：`git --version`

### 2. Python
- 推荐 Anaconda/Miniconda：https://docs.conda.io/en/latest/miniconda.html
- 验证：`python --version`

### 3. VSCode
- 下载：https://code.visualstudio.com/
- 推荐插件：
  - Python
  - Remote-SSH（用于远程连接服务器）
  - GitLens

### 4. LabelMe（图像标注）
```bash
pip install labelme
```
- 验证：`labelme --version`

---

## 三、Python 库

### 创建虚拟环境（推荐）
```bash
conda create -n torch_cu121 python=3.10 -y
conda activate torch_cu121
```

### 安装 PyTorch（CUDA 12.1）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
- 验证：`python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`

### 安装其他依赖
```bash
pip install rasterio numpy tqdm pillow
```

### 验证清单
```bash
python -c "import rasterio; print('rasterio OK')"
python -c "import numpy; print('numpy OK')"
python -c "import tqdm; print('tqdm OK')"
python -c "from PIL import Image; print('pillow OK')"
python -c "import labelme; print('labelme OK')"
```

---

## 四、GitHub 配置

### 1. 配置 Git 用户信息
```bash
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱"
```

### 2. 克隆仓库
```bash
git clone https://github.com/bboyWang/tp0630.git
cd tp0630
```

### 3. 验证远程连接
```bash
git remote -v
git pull origin master
```

---

## 五、数据准备

数据文件较大（1504 个 512×512 TIFF，30 波段），**不通过 GitHub 同步**。

**数据传输方式（任选其一）：**
1. **局域网共享**：台式机共享 `E:\tp0630\` 文件夹，另一台机器映射网络驱动器
2. **移动硬盘拷贝**：直接复制整个 `tp0630` 文件夹
3. **SCP 传输**：从笔记本 `scp -r E:\tp0630 用户@10.3.x.x:E:\`

**数据目录结构（必须保持一致）：**
```
E:\tp0630\
├── tiff\                    # 30波段 TIFF 文件（1504个）
├── eiseg_ready\images\      # PNG 预览图 + JSON 标注
├── checkpoints_v7\          # v7 模型权重
├── checkpoints_v8\          # v8 模型权重
├── ready.txt                # 标注注册表
├── results_v8.txt           # 训练结果
├── uncertain_v7.txt         # v7 不确定性排名
└── uncertain_v8.txt         # v8 不确定性排名
```

---

## 六、可选工具

| 工具 | 用途 | 安装 |
|------|------|------|
| EISeg | 交互式图像标注 | https://github.com/PaddlePaddle/PaddleSeg |
| 7-Zip | 压缩/解压 | https://www.7-zip.org/ |
| Notepad++ | 文本编辑 | https://notepad-plus-plus.org/ |

---

## 七、验证安装

运行以下命令确认环境就绪：

```bash
# 1. 进入项目目录
cd E:\tp0630  # 或你存放项目的路径

# 2. 激活虚拟环境
conda activate torch_cu121

# 3. 验证 PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 4. 验证数据加载（需数据已就位）
python -c "from dataset import WetlandDataset; ds = WetlandDataset('E:\\tp0630', patch_size=256); print(f'Dataset: {len(ds)} samples')"

# 5. 验证模型
python -c "from model import UNet; m = UNet(in_channels=26, num_classes=3); print(f'Model: {sum(p.numel() for p in m.parameters()):,} params')"
```

---

## 八、常见问题

### Q1: torch.cuda.is_available() 返回 False
- 检查 NVIDIA 驱动是否安装
- 检查 CUDA 版本是否与 PyTorch 匹配（本项目用 CUDA 12.1）

### Q2: rasterio 安装失败
```bash
# 尝试用 conda 安装
conda install -c conda-forge rasterio
```

### Q3: Git 推送失败（权限）
- 检查 GitHub 凭据：`git config --global credential.helper manager`
- 或使用 SSH 密钥替代 HTTPS

---

*最后更新: 2026-07-30*
