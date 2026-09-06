#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据集与基准基础设施 —— 目标检测对抗贴片基准（Benchmark）。

职责（对应基准文档 3.1）：
1. 确定性生成轻量合成目标检测场景（离线、无网络依赖），缓存到 data/benchmark.npz；
2. 训练 / 缓存统一检测模型 TinyDetector（单阶段 anchor-free，网格 16x16 @ 128px，
   通道 16-32，在 attack_train + defense_train 共 280 张图上训练 12 epochs，seed 42），
   权重缓存到 data/detector.pt；
3. 提供裁判端贴片渲染协议（apply_patch / apply_occlusion：位置 / 缩放 / 旋转 / 颜色抖动），
   供攻击方 EOT 优化与评测端渲染使用（对齐赛题 §12/§19）。

数据划分（seed=42 确定性生成，顺序切分）：
    attack_train = 120 / attack_val = 30 / defense_train = 160 / defense_val = 60 / hidden_test = 80
    （defense_train / defense_val 为 clean 图；其 patched 一半由 test.py 用基线攻击池渲染生成）

统一样本格式（Sample）：
    {"image_id": int,
     "image":    [H, W, 3] uint8，像素取值 [0,255]，H = W = 128，
     "annotations": [{"bbox": [x1, y1, x2, y2] 像素坐标, "category_id": 1(person)}]}

用法：
    python dataset.py            # 生成/加载基准数据并打印规模
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 基准常量（与文档 3.1 一致）
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 128                # 图像边长（正方形）
SEED = 42                       # 全局随机种子
NUM_OBJECTS_MAX = 4             # 每图最多目标数
TARGET_COCO_ID = 1              # COCO 类别 id：person
TARGET_NAME = "person"

PATCH_SIZE = 32                 # 攻击贴片边长：输出 (3, 32, 32) float [0,1]
MAX_PATCH_AREA_RATIO = 0.80     # 贴片面积 / 目标框面积上限（赛题建议 0.20；本轻量基准为演示攻击效果上调）
ROTATION_RANGE = (-15, 15)      # 旋转角度范围（度）
SCALE_RANGE = (0.85, 1.15)      # 缩放抖动范围
POSITION_INNER_RATIO = 0.60     # 贴片中心限制在目标框中央 60% 区域
COLOR_JITTER = 0.10             # 颜色抖动幅度

DET_SCORE_THRESH = 0.25         # 检测置信度阈值 τ_det
NMS_IOU = 0.45                  # NMS IoU 阈值
IOU_THRESH = 0.50               # Clean Detectable 判定 IoU 阈值

# 数据划分大小
SPLIT_SIZES = {
    "attack_train": 120,
    "attack_val": 30,
    "defense_train": 160,
    "defense_val": 60,
    "hidden_test": 80,
}
TOTAL_IMAGES = sum(SPLIT_SIZES.values())

CACHE_DIR = "data"


# =========================================================================== #
# 一、合成目标检测场景生成（离线兜底，确定性）
# =========================================================================== #
def _make_background(size: int, rng, hard_negative: bool = True) -> np.ndarray:
    """自然背景：随机两色渐变 + 噪声，偶尔叠加 hard-negative 装饰块。"""
    c1 = rng.integers(40, 200, size=3)
    c2 = rng.integers(40, 200, size=3)
    yy, xx = np.mgrid[0:size, 0:size]
    t = (yy + xx) / (2.0 * size)
    bg = c1[None, None, :] * (1 - t[:, :, None]) + c2[None, None, :] * t[:, :, None]
    bg += rng.normal(0, 8, bg.shape)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    if hard_negative and rng.random() < 0.4:
        # 叠加非目标的装饰方块/条纹（模拟广告牌/logo，避免防御模型简单学“方块=攻击”）
        bh = int(rng.integers(size // 8, size // 3))
        bw = int(rng.integers(size // 8, size // 3))
        by = int(rng.integers(0, size - bh))
        bx = int(rng.integers(0, size - bw))
        mode = rng.choice(["solid", "stripes", "checker"])
        if mode == "solid":
            block = np.tile(rng.integers(0, 255, size=3), (bh, bw, 1))
        elif mode == "stripes":
            block = np.zeros((bh, bw, 3), np.uint8)
            c = rng.integers(0, 255, size=3)
            for i in range(0, bw, max(3, bw // 8)):
                block[:, i:i + 2] = c
        else:  # checker
            cell = max(2, bh // 6)
            c = rng.integers(0, 255, size=3)
            block = np.zeros((bh, bw, 3), np.uint8)
            for i in range(0, bh, cell):
                for j in range(0, bw, cell):
                    if ((i // cell) + (j // cell)) % 2 == 0:
                        block[i:i + cell, j:j + cell] = c
        bg[by:by + bh, bx:bx + bw] = block
    return bg


def _make_object(h: int, w: int, rng) -> np.ndarray:
    """目标对象：person 类矩形（基础色 [180,60,60]）+ 渐变 + 轻微纹理。"""
    base = np.array([180, 60, 60], dtype=np.float32)
    grad = np.linspace(0.75, 1.15, h)[:, None, None]
    rect = base[None, None, :] * grad
    rect = rect * np.linspace(0.95, 1.05, w)[None, :, None]
    rect += rng.normal(0, 6, rect.shape)
    return np.clip(rect, 0, 255).astype(np.uint8)


def _gen_one(size: int, rng) -> tuple:
    """生成一张合成图 + 标注。返回 (image HWC uint8, annotations[{bbox xyxy, category_id}])。"""
    img = _make_background(size, rng)
    n = int(rng.integers(1, 4))  # 每图 1-3 个目标
    anns = []
    for _ in range(n):
        scale = rng.uniform(0.18, 0.42)
        h = int(size * scale * 0.60)
        w = int(size * scale * 0.55)
        h = max(16, min(h, size - 4))
        w = max(16, min(w, size - 4))
        x = int(rng.integers(2, max(3, size - w - 2)))
        y = int(rng.integers(2, max(3, size - h - 2)))
        img[y:y + h, x:x + w] = _make_object(h, w, rng)
        anns.append({"bbox": [x, y, x + w, y + h], "category_id": TARGET_COCO_ID})
    return img, anns


def generate_benchmark(cache_dir: str = CACHE_DIR, seed: int = SEED,
                       sizes: dict | None = None, force: bool = False) -> dict:
    """确定性生成基准数据并缓存为 data/benchmark.npz（images + 标注数组）。"""
    sizes = dict(sizes or SPLIT_SIZES)
    path = os.path.join(cache_dir, "benchmark.npz")
    if not force and os.path.exists(path):
        return _load_benchmark_from_npz(path, sizes)

    rng = np.random.default_rng(seed)
    total = sum(sizes.values())
    images = np.zeros((total, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    boxes = np.zeros((total, NUM_OBJECTS_MAX, 4), dtype=np.float32)
    cids = np.zeros((total, NUM_OBJECTS_MAX), dtype=np.int64)
    counts = np.zeros(total, dtype=np.int64)
    for i in range(total):
        img, anns = _gen_one(IMAGE_SIZE, rng)
        images[i] = img
        for k, a in enumerate(anns[:NUM_OBJECTS_MAX]):
            boxes[i, k] = a["bbox"]
            cids[i, k] = a["category_id"]
        counts[i] = min(len(anns), NUM_OBJECTS_MAX)

    # 按顺序切分（确定性）
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, images=images, boxes=boxes, cids=cids, counts=counts,
                        split_sizes=np.array([sizes[k] for k in sizes], dtype=np.int64),
                        split_names=np.array(list(sizes.keys())))
    print(f"[dataset] 已生成基准数据缓存: {path}（{total} 张图像）")
    return _load_benchmark_from_npz(path, sizes)


def _load_benchmark_from_npz(path: str, sizes: dict) -> dict:
    d = np.load(path)
    images = d["images"]
    boxes = d["boxes"]
    cids = d["cids"]
    counts = d["counts"]
    out = {}
    start = 0
    for name, n in sizes.items():
        samples = []
        for i in range(start, start + n):
            anns = []
            for k in range(int(counts[i])):
                x1, y1, x2, y2 = [float(v) for v in boxes[i, k]]
                anns.append({"bbox": [x1, y1, x2, y2],
                             "category_id": int(cids[i, k])})
            samples.append({"image_id": i, "image": images[i], "annotations": anns})
        out[name] = samples
        start += n
    return out


def load_benchmark(cache_dir: str = CACHE_DIR, seed: int = SEED,
                   sizes: dict | None = None, force: bool = False) -> dict:
    """加载基准数据，返回 {split_name: [Sample, ...]}。首次调用自动生成并落盘。"""
    sizes = dict(sizes or SPLIT_SIZES)
    return generate_benchmark(cache_dir=cache_dir, seed=seed, sizes=sizes, force=force)


# =========================================================================== #
# 二、统一检测模型 TinyDetector（单阶段 anchor-free，网格 16x16 @ 128px）
# =========================================================================== #
class _TinyNet(nn.Module):
    """轻量可微单阶段检测网络：3 个 stride-2 卷积块（128->64->32->16 特征图，网格 16x16，stride 8），
    每格预测 objectness(1) + 类别(K) + 框(4)，FCOS 风格 per-cell 回归。端到端可微，
    供攻击方梯度贴片优化（白盒前向 + 反传）。"""

    def __init__(self, num_classes: int, image_size: int = IMAGE_SIZE, device: str = "cpu"):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.stride = 8
        self.grid = image_size // self.stride  # 16
        self.device = device

        def conv_bn(ci, co, k=3, s=2, p=1):
            return nn.Sequential(
                nn.Conv2d(ci, co, k, s, p, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True))

        c = 16
        self.backbone = nn.Sequential(
            conv_bn(3, c), conv_bn(c, c * 2), conv_bn(c * 2, c * 2)).to(device)
        # FCOS 风格 3x3 检测头：每个格的 objectness 依赖其 3x3 邻域特征（引入局部上下文，
        # 使贴片能影响更大范围的检测响应；FCOS 原版即使用 3x3 head 卷积）
        self.head_obj = nn.Conv2d(c * 2, 1, 3, padding=1).to(device)
        self.head_cls = nn.Conv2d(c * 2, num_classes, 3, padding=1).to(device)
        self.head_box = nn.Conv2d(c * 2, 4, 3, padding=1).to(device)
        self.target_classes_local = {}  # coco id -> local idx

    def forward(self, x):
        feat = self.backbone(x)
        obj = self.head_obj(feat)          # (B,1,G,G)
        cls = self.head_cls(feat)          # (B,K,G,G)
        raw = self.head_box(feat)          # (B,4,G,G)
        dist = torch.exp(raw.clamp(-6.0, 6.0))  # 正距离
        G = self.grid
        coords = (torch.arange(G, device=x.device).float() + 0.5) * self.stride
        cx_grid = coords.view(1, G).expand(G, G)
        cy_grid = coords.view(G, 1).expand(G, G)
        l, t, r, b = dist[:, 0], dist[:, 1], dist[:, 2], dist[:, 3]
        x1 = cx_grid.unsqueeze(0) - l
        y1 = cy_grid.unsqueeze(0) - t
        x2 = cx_grid.unsqueeze(0) + r
        y2 = cy_grid.unsqueeze(0) + b
        box = torch.stack([x1, y1, x2, y2], dim=1)  # (B,4,G,G) xyxy
        return obj, cls, box

    def compute_loss(self, x, targets):
        """targets: list[dict{boxes(xyxy), labels(long, coco id)}]。"""
        B = x.shape[0]
        obj, cls, box = self.forward(x)
        G = self.grid
        device = x.device
        H = self.image_size
        obj_tgt = torch.zeros(B, 1, G, G, device=device)
        cls_tgt = torch.zeros(B, self.num_classes, G, G, device=device)
        box_tgt = torch.zeros(B, 4, G, G, device=device)
        pos_mask = torch.zeros(B, 1, G, G, device=device)
        coords = (torch.arange(G, device=device).float() + 0.5) * self.stride

        for b, t in enumerate(targets):
            boxes = t["boxes"]
            labels = t["labels"]
            if len(boxes) == 0:
                continue
            for bx, lb in zip(boxes, labels):
                x1, y1, x2, y2 = [float(v) for v in bx]
                gx_in = (coords >= x1) & (coords <= x2)
                gy_in = (coords >= y1) & (coords <= y2)
                gi_idx = torch.nonzero(gy_in, as_tuple=True)[0]
                gj_idx = torch.nonzero(gx_in, as_tuple=True)[0]
                if len(gi_idx) == 0 or len(gj_idx) == 0:
                    gi = int(min(max(((y1 + y2) / 2) / self.stride, 0), G - 1))
                    gj = int(min(max(((x1 + x2) / 2) / self.stride, 0), G - 1))
                    obj_tgt[b, 0, gi, gj] = 1.0
                    pos_mask[b, 0, gi, gj] = 1.0
                    box_tgt[b, :, gi, gj] = torch.tensor([x1, y1, x2, y2], device=device)
                    if int(lb) in self.target_classes_local:
                        cls_tgt[b, self.target_classes_local[int(lb)], gi, gj] = 1.0
                    continue
                for gi in gi_idx:
                    for gj in gj_idx:
                        obj_tgt[b, 0, gi, gj] = 1.0
                        pos_mask[b, 0, gi, gj] = 1.0
                        box_tgt[b, :, gi, gj] = torch.tensor([x1, y1, x2, y2], device=device)
                        if int(lb) in self.target_classes_local:
                            cls_tgt[b, self.target_classes_local[int(lb)], gi, gj] = 1.0

        pos_count = pos_mask.sum().clamp(min=1.0)
        neg_count = (B * G * G - pos_count).clamp(min=1.0)
        pw = (neg_count / pos_count).clamp(max=50.0)
        obj_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            obj, obj_tgt, reduction="mean", pos_weight=pw)
        pos = pos_mask.expand_as(cls_tgt)
        cls_loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            cls, cls_tgt, reduction="none") * pos).sum() / (pos.sum() + 1e-6)
        box_norm = box / H
        box_tgt_norm = box_tgt / H
        box_loss = (torch.nn.functional.l1_loss(box_norm, box_tgt_norm, reduction="none")
                    * pos[:, 0:1]).sum() / (pos.sum() + 1e-6)
        return {"loss_obj": obj_loss, "loss_cls": cls_loss * 2.0, "loss_box": box_loss * 5.0}


class TinyDetector:
    """统一检测模型（Model-A）封装：predict / loss / hiding_loss。冻结权重，仅放行输入梯度。"""

    name = "tiny_detector"

    def __init__(self, target_class_ids, device="cpu", score_thresh=DET_SCORE_THRESH,
                 nms_iou=NMS_IOU, input_size=IMAGE_SIZE, weights_path=None):
        self.target_class_ids = list(target_class_ids)
        self.coco_to_local = {cid: i for i, cid in enumerate(self.target_class_ids)}
        self.device = device
        self.score_thresh = score_thresh
        self.nms_iou = nms_iou
        self.input_size = input_size
        self.weights_path = weights_path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _TinyNet(num_classes=len(self.target_class_ids),
                                   image_size=self.input_size, device=self.device)
            self._model.target_classes_local = self.coco_to_local
            if self.weights_path and os.path.exists(self.weights_path):
                sd = torch.load(self.weights_path, map_location=self.device)
                self._model.load_state_dict(sd)
                print(f"[detector] TinyDetector 已加载权重 {self.weights_path}")
            else:
                raise FileNotFoundError(
                    f"检测器权重缺失: {self.weights_path}。请先运行 python test.py --stage attack "
                    f"或 python dataset.py --train-detector")
        return self._model

    # ---- 输入转换 ----
    def _to_tensor(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        out = []
        for img in images:
            if isinstance(img, np.ndarray):
                arr = img.astype(np.float32)
                if arr.max() > 1.5:
                    arr = arr / 255.0
                if arr.ndim == 3 and arr.shape[-1] in (1, 3):  # HWC -> CHW
                    arr = np.transpose(arr, (2, 0, 1))
                t = torch.from_numpy(np.ascontiguousarray(arr))
            else:
                t = img.float()
                if t.ndim == 3 and t.shape[-1] in (1, 3):
                    t = t.permute(2, 0, 1)
            out.append(t.to(self.device))
        return out

    @staticmethod
    def _targets_to_tensor(targets, device):
        out = []
        for t in targets:
            boxes = t["boxes"]
            if isinstance(boxes, np.ndarray):
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
            else:
                boxes = boxes.clone().float()
            labels = t["labels"]
            if isinstance(labels, np.ndarray):
                labels = torch.as_tensor(labels, dtype=torch.long)
            else:
                labels = labels.clone().long()
            out.append({"boxes": boxes.to(device), "labels": labels.to(device)})
        return out

    # ---- 推理 ----
    def predict(self, images):
        """返回 list[dict{boxes(N,4) xyxy, labels(N,), scores(N,)}]（numpy）。"""
        from torchvision.ops import nms
        m = self.model
        m.backbone.eval()
        tensors = self._to_tensor(images)
        results = []
        with torch.no_grad():
            for x in tensors:
                x = x.unsqueeze(0)
                obj, cls, box = m.forward(x)
                obj_p = torch.sigmoid(obj[0, 0])          # (G,G)
                cls_logits = cls[0]                        # (K,G,G)
                box_xyxy = box[0]                          # (4,G,G)
                mask = obj_p >= self.score_thresh
                if mask.sum() == 0:
                    results.append({"boxes": np.zeros((0, 4), np.float32),
                                    "labels": np.zeros((0,), np.int64),
                                    "scores": np.zeros((0,), np.float32)})
                    continue
                idx = mask.flatten().nonzero(as_tuple=True)[0]
                gi = idx // m.grid
                gj = idx % m.grid
                scores = obj_p[gi, gj]
                boxes = box_xyxy[:, gi, gj].T  # (N,4)
                labels_local = cls_logits[:, gi, gj].argmax(0)
                labels = torch.tensor([self.target_class_ids[int(l)] for l in labels_local],
                                      device=x.device)
                keep = nms(boxes, scores, self.nms_iou)
                results.append({
                    "boxes": boxes[keep].cpu().numpy().astype(np.float32),
                    "labels": labels[keep].cpu().numpy().astype(np.int64),
                    "scores": scores[keep].cpu().numpy().astype(np.float32),
                })
        return results

    # ---- 训练损失（攻击方梯度优化用，保留输入计算图） ----
    def loss(self, images, targets):
        m = self.model
        m.backbone.train()
        tensors = self._to_tensor(images)
        tensors = [t if t.requires_grad else t.detach().requires_grad_(True) for t in tensors]
        tgt = self._targets_to_tensor(targets, self.device)
        x = torch.stack(tensors)
        losses = m.compute_loss(x, tgt)
        return tensors, losses

    def hiding_loss(self, images, targets, w_box: float = 2.0):
        """直接隐藏目标损失（交付 GradientPatchAttack 的目标函数，基准文档 2.1 一致）：
        最小化目标框内网格单元 sigmoid(objectness) 的均值（压制检测置信度），
        并叠加 w_box 倍的框回归破坏项（把预测框推离 GT 框，使 IoU < 0.5）。
        攻击方最小化该标量：非饱和、梯度恒定，比纯检测训练损失更有效。"""
        m = self.model
        m.backbone.eval()  # 稳定 BatchNorm，减少梯度噪声
        tensors = self._to_tensor(images)
        tensors = [t if t.requires_grad else t.detach().requires_grad_(True) for t in tensors]
        x = torch.stack(tensors)
        obj, _, box = m.forward(x)
        G = m.grid
        device = x.device
        coords = (torch.arange(G, device=device).float() + 0.5) * m.stride
        obj_term = x.new_zeros(())
        box_disrupt = x.new_zeros(())
        n = 0
        for b, t in enumerate(targets):
            for bx in t["boxes"]:
                x1, y1, x2, y2 = [float(v) for v in bx]
                gx_in = (coords >= x1) & (coords <= x2)
                gy_in = (coords >= y1) & (coords <= y2)
                if gx_in.sum() == 0 or gy_in.sum() == 0:
                    continue
                region = obj[b, 0][gy_in][:, gx_in].reshape(-1)     # (ny*nx,) logits
                obj_term = obj_term + torch.sigmoid(region).mean()  # 压制检测置信度
                box_region = box[b, :, gy_in][:, :, gx_in]          # (4,ny,nx) xyxy
                gt = torch.tensor([x1, y1, x2, y2], device=device).view(4, 1, 1)
                box_disrupt = box_disrupt - (box_region - gt).abs().mean()  # 推离 GT 框
                n += 1
        loss = (obj_term + w_box * box_disrupt) / max(1, n)
        return tensors, loss


def train_tiny_detector(images, annotations, target_class_ids, image_size=IMAGE_SIZE,
                        epochs=12, lr=1e-3, batch_size=8, device="cpu",
                        save_path=None, seed=SEED):
    """在合成数据上训练 TinyDetector。images: list[HWC uint8]; annotations: list[list[{bbox,cid}]]。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = _TinyNet(num_classes=len(target_class_ids), image_size=image_size, device=device)
    net.target_classes_local = {cid: i for i, cid in enumerate(target_class_ids)}

    class _DS(torch.utils.data.Dataset):
        def __init__(s, imgs, anns):
            s.imgs, s.anns = imgs, anns

        def __len__(s):
            return len(s.imgs)

        def __getitem__(s, i):
            arr = s.imgs[i].astype(np.float32) / 255.0
            if arr.shape[-1] in (1, 3):
                arr = np.transpose(arr, (2, 0, 1))
            t = torch.from_numpy(np.ascontiguousarray(arr))
            boxes = torch.tensor([a["bbox"] for a in s.anns[i]], dtype=torch.float32) \
                if s.anns[i] else torch.zeros((0, 4))
            labels = torch.tensor([a["category_id"] for a in s.anns[i]], dtype=torch.long) \
                if s.anns[i] else torch.zeros((0,), dtype=torch.long)
            return t, {"boxes": boxes, "labels": labels}

    dl = torch.utils.data.DataLoader(
        _DS(images, annotations), batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=False,
        collate_fn=lambda batch: (torch.stack([b[0] for b in batch]), [b[1] for b in batch]))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    print(f"[tiny_detector] 训练 {epochs} epochs, {len(images)} 样本, {len(target_class_ids)} 类 ...")
    for ep in range(epochs):
        tot = 0.0
        for x, tgt in dl:
            x = x.to(device)
            tgt = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in tgt]
            losses = net.compute_loss(x, tgt)
            loss = sum(losses.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * x.shape[0]
        if (ep + 1) % max(1, epochs // 4) == 0 or ep == 0:
            print(f"[tiny_detector] epoch {ep + 1}/{epochs}  loss={tot / max(1, len(images)):.4f}")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(net.state_dict(), save_path)
        print(f"[tiny_detector] 权重已保存 {save_path}")
    return net


def build_detector(cache_dir: str = CACHE_DIR, force_train: bool = False,
                   device: str = "cpu") -> TinyDetector:
    """构建统一检测模型：优先加载缓存权重，缺失时训练并缓存。"""
    weights_path = os.path.join(cache_dir, "detector.pt")
    bundle = load_benchmark(cache_dir=cache_dir)
    if not force_train and os.path.exists(weights_path):
        return TinyDetector(target_class_ids=[TARGET_COCO_ID], device=device,
                            weights_path=weights_path)
    # 训练数据 = attack_train + defense_train 的 clean 图（与文档一致，共 280 张）
    train_samples = bundle["attack_train"] + bundle["defense_train"]
    images = [s["image"] for s in train_samples]
    annotations = [s["annotations"] for s in train_samples]
    train_tiny_detector(images, annotations, target_class_ids=[TARGET_COCO_ID],
                        image_size=IMAGE_SIZE, epochs=12, lr=1e-3, batch_size=8,
                        device=device, save_path=weights_path, seed=SEED)
    return TinyDetector(target_class_ids=[TARGET_COCO_ID], device=device,
                        weights_path=weights_path)


# =========================================================================== #
# 三、裁判端贴片渲染协议（对齐赛题 §12/§19；本模块不提供给防御方）
# =========================================================================== #
def sample_params(bbox, rng) -> dict:
    """采样贴片渲染参数。bbox: [x1,y1,x2,y2]（图像坐标）。"""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    bbox_area = bw * bh
    base_side = np.sqrt(bbox_area * MAX_PATCH_AREA_RATIO)  # 贴片面积 = 50% 目标框面积
    scale = float(rng.uniform(*SCALE_RANGE))
    side = max(8, int(round(base_side * scale)))
    side = min(side, int(min(bw, bh)))  # 不超过目标框短边
    angle = np.radians(float(rng.uniform(*ROTATION_RANGE)))
    inner = POSITION_INNER_RATIO
    cx_min = x1 + bw * (1 - inner) / 2
    cx_max = x2 - bw * (1 - inner) / 2
    cy_min = y1 + bh * (1 - inner) / 2
    cy_max = y2 - bh * (1 - inner) / 2
    cx = float(rng.uniform(cx_min, cx_max))
    cy = float(rng.uniform(cy_min, cy_max))
    color_factor = float(rng.uniform(1 - COLOR_JITTER, 1 + COLOR_JITTER))
    return {"side": side, "angle": angle, "cx": cx, "cy": cy,
            "color_factor": color_factor, "bbox": [x1, y1, x2, y2]}


def occlusion_patch(kind: str, side: int, rng) -> np.ndarray:
    """生成自然遮挡贴片 (3,side,side) float [0,1]。"""
    if kind == "gray":
        p = np.full((side, side, 3), 128, dtype=np.uint8)
    elif kind == "black":
        p = np.zeros((side, side, 3), dtype=np.uint8)
    elif kind == "white":
        p = np.full((side, side, 3), 255, dtype=np.uint8)
    elif kind == "noise":
        p = rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)
    elif kind == "texture":
        p = np.zeros((side, side, 3), dtype=np.uint8)
        c = rng.integers(0, 256, size=3)
        cell = max(2, side // 8)
        for i in range(0, side, cell):
            for j in range(0, side, cell):
                if ((i // cell) + (j // cell)) % 2 == 0:
                    p[i:i + cell, j:j + cell] = c
        p = p + rng.integers(-10, 10, p.shape)
        p = np.clip(p, 0, 255).astype(np.uint8)
    else:
        p = np.full((side, side, 3), 128, dtype=np.uint8)
    return np.transpose(p.astype(np.float32) / 255.0, (2, 0, 1))  # CHW float [0,1]


OCC_POOL = ["gray", "black", "white", "noise", "texture"]


def render_patch_torch(image, patch, params):
    """可微贴片渲染：image (1,3,H,W) float [0,1]，patch (3,ps,ps) float [0,1]（可微），
    返回 (1,3,H,W) float [0,1]，与 patch 保持计算图连接。"""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    B = image.shape[0]
    device = image.device
    side = int(params["side"])
    angle = float(params["angle"])
    cx, cy = float(params["cx"]), float(params["cy"])
    cf = float(params["color_factor"])
    img_size = image.shape[-1]

    p = patch.unsqueeze(0)  # (1,3,ps,ps)
    if side != patch.shape[-1]:
        p = F.interpolate(p, size=(side, side), mode="bilinear", align_corners=False)
    p = (p * cf).clamp(0, 1)

    yy, xx = torch.meshgrid(
        torch.arange(img_size, dtype=torch.float32, device=device),
        torch.arange(img_size, dtype=torch.float32, device=device),
        indexing="ij")
    dx = xx - cx
    dy = yy - cy
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    px = cos_a * dx + sin_a * dy
    py = -sin_a * dx + cos_a * dy
    half = side / 2.0
    u = px / half
    v = py / half
    grid = torch.stack([u, v], dim=-1).unsqueeze(0)  # (1,H,W,2)
    sampled = F.grid_sample(p, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask_u = (u.abs() <= 1.0).float()
    mask_v = (v.abs() <= 1.0).float()
    mask = (mask_u * mask_v).unsqueeze(0)  # (1,1,H,W)
    out = image * (1 - mask) + sampled * mask
    return out.clamp(0, 1)


def render_patch(image_hwc, patch, params) -> np.ndarray:
    """渲染贴片到 HWC uint8 图像 -> HWC uint8（评测用，无梯度）。"""
    img_t = torch.from_numpy(np.ascontiguousarray(
        image_hwc.astype(np.float32) / 255.0).transpose(2, 0, 1)).float().unsqueeze(0)
    p_t = torch.from_numpy(np.ascontiguousarray(patch.astype(np.float32))).float()
    with torch.no_grad():
        out = render_patch_torch(img_t, p_t, params)
    arr = out[0].permute(1, 2, 0).cpu().numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def apply_patch(image_hwc, bbox, patch, rng):
    """裁判端：采样参数 + 渲染攻击贴片。返回 (patched_hwc, params)。"""
    params = sample_params(bbox, rng)
    out = render_patch(image_hwc, patch, params)
    return out, params


def apply_occlusion(image_hwc, bbox, params, rng):
    """用相同参数渲染自然遮挡贴片（§19 公平扣减遮挡）。返回 (patched_hwc, kind)。"""
    kind = str(rng.choice(OCC_POOL))
    occ = occlusion_patch(kind, int(params["side"]), rng)
    out = render_patch(image_hwc, occ, params)
    return out, kind


# =========================================================================== #
# 四、攻击环境与贴片工具
# =========================================================================== #
class AttackEnv:
    """攻击方环境：公开检测器（白盒）+ 可微渲染 + 配置（对齐赛题 §15）。"""

    def __init__(self, detector, device="cpu", patch_size=PATCH_SIZE, image_size=IMAGE_SIZE):
        self.detector = detector
        self.device = device
        self.patch_size = patch_size
        self.image_size = image_size

    def sample_params(self, bbox, rng):
        return sample_params(bbox, rng)

    def render_patch_torch(self, image, patch, params):
        return render_patch_torch(image, patch, params)


def init_patch(mode: str, size: int, rng) -> np.ndarray:
    """初始化贴片 (3,size,size) float [0,1]：noise / gray / checker。"""
    if mode == "gray":
        return np.full((3, size, size), 0.5, dtype=np.float32)
    if mode == "checker":
        p = np.zeros((3, size, size), dtype=np.float32)
        cell = max(2, size // 8)
        c0, c1 = rng.random(3), rng.random(3)
        for i in range(0, size, cell):
            for j in range(0, size, cell):
                c = c0 if ((i // cell) + (j // cell)) % 2 == 0 else c1
                p[:, i:i + cell, j:j + cell] = c[:, None, None]
        return p
    return rng.random((3, size, size), dtype=np.float32)  # noise


def save_patch(patch: np.ndarray, path: str) -> None:
    if not path.endswith(".npy"):
        path = path + ".npy"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, np.asarray(patch, dtype=np.float32))


def load_patch(path: str) -> np.ndarray:
    if not os.path.exists(path) and not path.endswith(".npy"):
        alt = path + ".npy"
        if os.path.exists(alt):
            path = alt
    return np.load(path)


# =========================================================================== #
# 命令行入口
# =========================================================================== #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="目标检测对抗贴片基准：数据集与检测器")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--train-detector", action="store_true", help="强制重训检测器")
    ap.add_argument("--force", action="store_true", help="强制重新生成数据")
    args = ap.parse_args()

    bundle = load_benchmark(cache_dir=args.cache_dir, force=args.force)
    for name, samples in bundle.items():
        print(f"[dataset] {name}: {len(samples)} 张图像")
    if args.train_detector:
        det = build_detector(cache_dir=args.cache_dir, force_train=True)
        print(f"[dataset] 检测器就绪: {det.name}")
