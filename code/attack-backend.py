#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""攻击代码 —— Baseline：梯度优化通用对抗贴片（GradientPatchAttack）。

原理（对应基准文档 2.1）：采用 DPatch / EOT 思想，用公开检测器 Model-A（TinyDetector，
白盒前向 + 反传）训练一张 (3,32,32) 的通用对抗贴片：每步对目标框随机采样渲染参数
（位置 / 缩放 / 旋转 / 颜色抖动，EOT），可微渲染后送入检测器，最小化"隐藏目标损失"
（压低目标框内 objectness logit 的 logsumexp + 扰乱框回归）+ 全变分（TV）正则，
Adam 更新贴片并裁剪到 [0,1]。训练完成的贴片由裁判端统一渲染到隐藏图上评测。

接口（对齐赛题 §15-§18）：
    attack(env, task) -> {"patch": ndarray (3, 32, 32) float32，取值 [0,1]}

task = {
    "images":       list[HWC uint8]  攻击优化训练图像
    "annotations":  list[list[{bbox, category_id}]]  每图目标框
    "target_class": int   目标 COCO 类别 id（1 = person）
    "patch_size":   int   贴片边长（32）
    "max_patch_area_ratio": float
    "detector":     env.detector  公开 Model-A（predict / loss / hiding_loss）
}

用法：
    python attack.py                  # 读取基准数据，输出 data/attack_patch.npy
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from dataset import (AttackEnv, PATCH_SIZE, TARGET_COCO_ID, build_detector,
                     init_patch, load_benchmark, save_patch)

DEVICE = "cpu"


# --------------------------------------------------------------------------- #
# 梯度优化通用对抗贴片（交付 Baseline）
# --------------------------------------------------------------------------- #
def _total_variation(p: torch.Tensor) -> torch.Tensor:
    """p: (3,H,W) tensor。全变分（平滑 / 可打印性）正则。"""
    return (p[:, 1:, :] - p[:, :-1, :]).abs().mean() + (p[:, :, 1:] - p[:, :, :-1]).abs().mean()


# [IMPORTANT] Replace this line with real attack/defense code.

# --------------------------------------------------------------------------- #
# 基线攻击池（A0，用于防御训练数据与全对阵矩阵；非交付 Baseline）
# --------------------------------------------------------------------------- #
class RandomNoiseAttack:
    """A0-1 随机噪声贴片（最低攻击基线，赛题 §45 / §53）。"""

    name = "random_noise"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def __call__(self, env, task: dict) -> dict:
        ps = int(task["patch_size"])
        rng = np.random.default_rng(self.seed)
        return {"patch": rng.random((3, ps, ps), dtype=np.float32)}


class CheckerboardAttack:
    """A0-2 棋盘高对比纹理贴片（非优化高频纹理基线，赛题 §45）。"""

    name = "checkerboard"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def __call__(self, env, task: dict) -> dict:
        ps = int(task["patch_size"])
        rng = np.random.default_rng(self.seed)
        return {"patch": init_patch("checker", ps, rng)}


class GrayPatchAttack:
    """纯遮挡贴片（gray/black/white），用于对照"纯遮挡"基线——其 AttackGain 应接近 0，
    验证 AttackGain 公式能正确扣减自然遮挡影响（赛题 §19 / §22）。"""

    name = "gray_patch"

    def __init__(self, color: str = "gray", seed: int = 42):
        self.color = color
        self.seed = seed

    def __call__(self, env, task: dict) -> dict:
        ps = int(task["patch_size"])
        if self.color == "gray":
            p = np.full((3, ps, ps), 0.5, dtype=np.float32)
        elif self.color == "black":
            p = np.zeros((3, ps, ps), dtype=np.float32)
        elif self.color == "white":
            p = np.ones((3, ps, ps), dtype=np.float32)
        else:
            rng = np.random.default_rng(self.seed)
            p = rng.random((3, ps, ps), dtype=np.float32)
        return {"patch": p}


# --------------------------------------------------------------------------- #
# 攻击方法注册表（test.py / defense 训练数据复用）
# --------------------------------------------------------------------------- #
ATTACKS = {
    "random_noise": RandomNoiseAttack,
    "checkerboard": CheckerboardAttack,
    "gray_patch": GrayPatchAttack,
    "gradient_patch": GradientPatchAttack,
}


def attack(env, task: dict, name: str = "gradient_patch", **kwargs) -> dict:
    """统一攻击入口：默认使用交付 Baseline GradientPatchAttack。"""
    return ATTACKS[name](**kwargs)(env, task)


def validate_patch(result: dict, patch_size: int = PATCH_SIZE) -> tuple:
    """攻击合法性检查（赛题 §18）：尺寸 (3,32,32)、有限、像素 [0,1]。"""
    msgs = []
    if not isinstance(result, dict) or "patch" not in result:
        return False, None, ["输出必须为 dict 且包含 'patch' 字段"]
    patch = result["patch"]
    if isinstance(patch, torch.Tensor):
        patch = patch.detach().cpu().numpy()
    patch = np.asarray(patch)
    if patch.ndim != 3:
        return False, None, [f"patch 维度应为 3, 实际 {patch.ndim}"]
    if patch.shape[-1] == 3 and patch.shape[0] != 3:
        patch = np.transpose(patch, (2, 0, 1))
    if patch.shape[0] != 3:
        return False, None, [f"patch 通道数应为 3, 实际 shape {patch.shape}"]
    if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
        return False, None, [f"patch 尺寸应为 (3,{patch_size},{patch_size}), 实际 {patch.shape}"]
    if not np.isfinite(patch).all():
        msgs.append("警告: patch 含非有限值, 已替换为 0")
        patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)
    patch = patch.astype(np.float32)
    if patch.max() > 1.5:
        msgs.append("警告: patch 似乎为 [0,255], 已归一化到 [0,1]")
        patch = patch / 255.0
    patch = np.clip(patch, 0.0, 1.0)
    return True, patch, msgs


def build_task(env: AttackEnv, bundle: dict) -> dict:
    """从基准数据构建攻击 task（对齐赛题 §15）。"""
    images = [s["image"] for s in bundle["attack_train"]]
    annotations = [[{"bbox": a["bbox"], "category_id": int(a["category_id"])}
                    for a in s["annotations"]] for s in bundle["attack_train"]]
    return {
        "images": images,
        "annotations": annotations,
        "target_class": TARGET_COCO_ID,
        "patch_size": PATCH_SIZE,
        "max_patch_area_ratio": None,  # 渲染协议统一由 dataset.sample_params 控制
        "detector": env.detector,
    }


# --------------------------------------------------------------------------- #
# 命令行入口
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="梯度优化通用对抗贴片攻击（Baseline）")
    parser.add_argument("--iters", type=int, default=250, help="优化迭代步数")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam 学习率")
    parser.add_argument("--batch-size", type=int, default=8, help="每步训练图数（EOT）")
    parser.add_argument("--tv-weight", type=float, default=0.01, help="全变分正则权重")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", default="data/attack_patch.npy", help="输出贴片路径")
    args = parser.parse_args()

    bundle = load_benchmark()
    det = build_detector()
    env = AttackEnv(det, device=DEVICE, patch_size=PATCH_SIZE)
    task = build_task(env, bundle)

    atk = GradientPatchAttack(iters=args.iters, lr=args.lr, batch_size=args.batch_size,
                              tv_weight=args.tv_weight, seed=args.seed)
    out = atk(env, task)
    ok, patch, msgs = validate_patch(out)
    for m in msgs:
        print(f"[attack] {m}")
    if not ok:
        raise SystemExit(f"[attack] 贴片不合法: {msgs}")
    save_patch(patch, args.output)
    print(f"[attack] GradientPatchAttack 完成，贴片 {patch.shape} 已保存: {args.output} "
          f"(值域 [{patch.min():.3f}, {patch.max():.3f}])")
