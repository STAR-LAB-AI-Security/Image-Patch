#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""防御代码 —— Baseline：ResNet18Classifier（ResNet18 预训练贴片检测二分类器）。

原理（对应基准文档 2.2）：防御方训练一个独立的图像二分类器（0 = 正常图片，
1 = 含对抗贴片图片），不修改、不替换目标检测模型。ResNet18Classifier 使用
ImageNet 预训练的 ResNet18 作为骨干（材料 D0-1 官方防御基线），将末层全连接
替换为单 logit 输出，以二分类交叉熵（BCEWithLogits）在"50% clean + 50% 基线
攻击池渲染的 patched"训练数据上微调；推理时对输入图像缩放、归一化后前向，
sigmoid 得到"含贴片概率"。防御方看不到攻击方最终提交的贴片（隐藏阶段未知贴片），
预训练特征使模型能泛化到训练阶段未见过的贴片。

接口：
    fit(denv, clean_images, patched_images, cfg) -> DefenseModel
    DefenseModel.predict(image: HWC uint8) -> float（patch_probability in [0,1]）

无网络回退：首次运行需从 PyTorch CDN 下载 ImageNet 预训练权重（约 45MB，缓存于
~/.cache/torch）；若下载失败，自动回退随机初始化并给出警告（性能下降）。

用法：
    python defense.py                  # 读取 data/defense_train_data.npz，输出模型与评测
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

DEVICE = "cpu"
INPUT_SIZE = 224          # ResNet18 标准输入边长（ImageNet 归一化）
EPOCHS = 8
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# --------------------------------------------------------------------------- #
# 防御模型（ResNet18Classifier）
# --------------------------------------------------------------------------- #
class DefenseModel:
    """已训练防御模型：predict(image) -> patch_probability in [0,1]。"""

    name = "resnet18_classifier"

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.net = None
        self.input_size = INPUT_SIZE

    def predict(self, image) -> float:
        x = self._preprocess(image)
        with torch.no_grad():
            logit = self.net(x.unsqueeze(0).to(self.device))
        prob = torch.sigmoid(logit).view(-1)[0].item()
        return float(np.clip(prob, 0.0, 1.0))

    def predict_many(self, images) -> np.ndarray:
        return np.asarray([self.predict(im) for im in images], dtype=np.float32)

    def _preprocess(self, image):
        arr = np.asarray(image).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 3:
            arr = np.transpose(arr, (2, 0, 1))
        t = torch.from_numpy(np.ascontiguousarray(arr))
        from torchvision.transforms import functional as T
        t = T.resize(t, [self.input_size, self.input_size])
        return (t - _IMAGENET_MEAN) / _IMAGENET_STD


def _build_resnet18(pretrained: bool = True) -> nn.Module:
    import torchvision
    if pretrained:
        try:
            net = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            print("[resnet18] ImageNet 预训练权重加载成功")
        except Exception as exc:
            print(f"[resnet18] 预训练权重下载/加载失败（{exc}），回退随机初始化")
            net = torchvision.models.resnet18(weights=None)
    else:
        net = torchvision.models.resnet18(weights=None)
    # 将末层全连接替换为 1 个 logit（sigmoid -> 含贴片概率）
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net


class ResNet18Classifier:
    """ResNet18 贴片检测二分类防御（交付 Baseline，材料 D0-1）。"""

    name = "resnet18_classifier"

    def __init__(self, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
                 lr: float = LR, weight_decay: float = WEIGHT_DECAY,
                 input_size: int = INPUT_SIZE, seed: int = 42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.input_size = input_size
        self.seed = seed

    def fit(self, denv, clean_images, patched_images, cfg=None) -> DefenseModel:
        """训练 ResNet18 二分类器。clean_images / patched_images: list[HWC uint8]。
        返回 DefenseModel 实例（predict(image) -> float）。"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        from torchvision.transforms import functional as T

        net = _build_resnet18(pretrained=True).to(DEVICE)

        class _DS(torch.utils.data.Dataset):
            def __init__(s, clean, patched):
                s.x, s.y = [], []
                for im in clean:
                    s.x.append(im); s.y.append(0)
                for im in patched:
                    s.x.append(im); s.y.append(1)

            def __len__(s):
                return len(s.x)

            def __getitem__(s, i):
                arr = s.x[i].astype(np.float32) / 255.0
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                if arr.shape[-1] == 3:
                    arr = np.transpose(arr, (2, 0, 1))
                t = torch.from_numpy(np.ascontiguousarray(arr))
                t = T.resize(t, [self.input_size, self.input_size])
                return (t - _IMAGENET_MEAN) / _IMAGENET_STD, s.y[i]

        dl = torch.utils.data.DataLoader(_DS(clean_images, patched_images),
                                         batch_size=self.batch_size, shuffle=True,
                                         num_workers=0, drop_last=False)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce = nn.BCEWithLogitsLoss()
        net.train()
        n = len(clean_images) + len(patched_images)
        print(f"[resnet18] 训练 {self.epochs} epochs, {len(clean_images)} clean + "
              f"{len(patched_images)} patched = {n} 样本 ...")
        for ep in range(self.epochs):
            tot, cnt = 0.0, 0
            for x, y in dl:
                x = x.to(DEVICE)
                y = y.float().to(DEVICE).view(-1, 1)
                loss = bce(net(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * x.shape[0]
                cnt += x.shape[0]
            if (ep + 1) % max(1, self.epochs // 4) == 0 or ep == 0:
                print(f"[resnet18] epoch {ep + 1}/{self.epochs}  loss={tot / max(1, cnt):.4f}")
        net.eval()

        m = DefenseModel(device=DEVICE)
        m.name = self.name
        m.net = net
        m.input_size = self.input_size
        return m


# --------------------------------------------------------------------------- #
# 命令行入口
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet18 贴片检测防御（Baseline）")
    parser.add_argument("--input", default="data/defense_train_data.npz",
                        help="防御训练数据（clean + patched 图像与标签，由 test.py 生成）")
    parser.add_argument("--model-output", default="data/defense_model.pt", help="模型保存路径")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    d = np.load(args.input)
    clean_imgs = [d["clean_images"][i] for i in range(d["clean_images"].shape[0])]
    patched_imgs = [d["patched_images"][i] for i in range(d["patched_images"].shape[0])]
    print(f"[defense] 加载防御训练数据: {len(clean_imgs)} clean + {len(patched_imgs)} patched")

    defense = ResNet18Classifier(epochs=args.epochs, batch_size=args.batch_size,
                                 lr=args.lr, seed=args.seed)
    model = defense.fit(None, clean_imgs, patched_imgs)

    os.makedirs(os.path.dirname(args.model_output) or ".", exist_ok=True)
    torch.save({"state_dict": model.net.state_dict(), "input_size": model.input_size},
               args.model_output)
    print(f"[defense] ResNet18Classifier 模型已保存: {args.model_output}")

    # 快速自检：训练集上的平衡准确率
    y_true = np.concatenate([np.zeros(len(clean_imgs)), np.ones(len(patched_imgs))])
    scores = model.predict_many(clean_imgs + patched_imgs)
    y_bin = (scores >= 0.5).astype(int)
    tp = int(np.sum((y_bin == 1) & (y_true == 1)))
    fn = int(np.sum((y_bin == 0) & (y_true == 1)))
    fp = int(np.sum((y_bin == 1) & (y_true == 0)))
    tn = int(np.sum((y_bin == 0) & (y_true == 0)))
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    print(f"[defense] 训练集自检: TPR={tpr:.3f} FPR={fpr:.3f} "
          f"BalAcc={(tpr + (1 - fpr)) / 2:.3f}")
