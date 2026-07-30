"""
轻量级 U-Net：30波段输入 → 3类输出（湿地/水体/湖泊）
专为 RTX 3050 4GB 优化：256×256 输入, batch=2

编码器: 30→32→64→128→256
解码器: 256→128→64→32→3
标签: 0=湿地, 1=水体, 2=湖泊, -1=忽略(背景)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """双卷积 + BN + ReLU"""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.conv(x))


class UNet(nn.Module):
    """
    轻量级 U-Net + 注意力门控

    编码器: 30→32→64→128→256
    解码器: 256→128→64→32→3
    """

    def __init__(self, in_channels=26, num_classes=3, base_ch=32, dropout=0.1):
        super().__init__()

        # === 编码器 ===
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, dropout)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, dropout)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, dropout)

        self.pool = nn.MaxPool2d(2)

        # === 瓶颈 ===
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8, dropout)

        # === 解码器 ===
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 4 + base_ch * 8, base_ch * 4, dropout)

        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 2 + base_ch * 4, base_ch * 2, dropout)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec2 = ConvBlock(base_ch + base_ch * 2, base_ch)

        # === 输出 ===
        self.out_conv = nn.Conv2d(base_ch, num_classes, 1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)       # (B, 32, H, W)
        e2 = self.enc2(self.pool(e1))   # (B, 64, H/2, W/2)
        e3 = self.enc3(self.pool(e2))   # (B, 128, H/4, W/4)
        e4 = self.enc4(self.pool(e3))   # (B, 256, H/8, W/8)

        # 瓶颈
        b = self.bottleneck(self.pool(e4))  # (B, 256, H/16, W/16)

        # 解码
        d4 = self.up4(b)        # (B, 128, H/8, W/8)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)       # (B, 64, H/4, W/4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)       # (B, 32, H/2, W/2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        # 上采样回原始尺寸
        out = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.out_conv(out)

        return out


def count_params(model):
    """统计参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    model = UNet(in_channels=30, num_classes=3)
    total, trainable = count_params(model)
    print(f"U-Net 参数量: {total:,} (训练: {trainable:,})")

    # 测试前向传播 + 显存估算
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    x = torch.randn(2, 30, 256, 256, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"输入: {x.shape} → 输出: {y.shape}")
    print(f"模型设备: {device}")

    if device == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"峰值显存: {mem:.1f} MB")
