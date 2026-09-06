#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试代码 —— 目标检测对抗贴片基准（Benchmark）指标计算。

流程（对齐赛题 §50/§52/§65）：
    攻击评测（run_attack.sh -> --stage attack）：
        生成 GradientPatchAttack 通用贴片 -> 合法性检查 -> 隐藏图渲染 adv/occ ->
        统一检测器推理 -> Recall / HideRate / AttackGain / ConfidenceDrop / AttackScore_base
    防御评测（run_defense.sh -> --stage defense）：
        基线攻击池渲染 clean/patched 训练数据 -> 训练 ResNet18Classifier ->
        验证集 TPR/FPR/BalAcc/AUROC（资格线）-> 隐藏阶段"未知贴片"对阵评测
    全对阵（--stage all）：
        A0 攻击池 × ResNet18Classifier 全矩阵 -> FinalAttackScore / FinalDefenseScore

指标公式（与基准文档 3.2 一致，Acc 类指标均基于统一检测器在隐藏集上的推理）：
    Recall_*        = N_correct / N_eligible（IoU>=0.5 & score>=0.25 & 类别一致）
    HideRate_adv    = 1 - Recall_adv； HideRate_occ = 1 - Recall_occ
    AttackGain      = clip(max(0, (Hide_adv-Hide_occ)/(1-Hide_occ+eps)), 0, 1)
    ConfidenceDrop  = mean(max(0, (Conf_clean-Conf_adv)/(Conf_clean+eps)))
    AttackScore_base= 0.85*AttackGain + 0.15*ConfidenceDrop
    TPR/FPR/TNR     = 标准二分类（阈值 0.5）； BalancedAccuracy = (TPR+TNR)/2
    AUROC           = 阈值无关判别能力（sklearn）
    DefenseScore_i  = 0.7*BalAcc_i + 0.3*AUROC_i
    EvasionRate     = 1 - TPR； AttackScore_{i,j} = base_i * (0.8 + 0.2*Evasion_{i,j})
    FinalAttackScore_i  = mean_j AttackScore_{i,j}
    FinalDefenseScore_j = mean_i DefenseScore_{i,j}

用法：
    python test.py --stage attack      # 攻击方 Baseline 评测（默认）
    python test.py --stage defense     # 防守方 Baseline 评测
    python test.py --stage all         # 完整攻防全对阵评测
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from attack import (ATTACKS, GradientPatchAttack, validate_patch, build_task)
from dataset import (AttackEnv, CACHE_DIR, IOU_THRESH, PATCH_SIZE, SEED, TARGET_COCO_ID,
                     apply_occlusion, apply_patch, build_detector, load_benchmark,
                     save_patch)
from defense import ResNet18Classifier

DEVICE = "cpu"
EPS = 1e-8

# --------------------------------------------------------------------------- #
# 指标函数（严格对齐基准文档 3.2 公式）
# --------------------------------------------------------------------------- #
def iou_xyxy(box_a, box_b) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a = (float(box_a[2]) - float(box_a[0])) * (float(box_a[3]) - float(box_a[1]))
    b = (float(box_b[2]) - float(box_b[0])) * (float(box_b[3]) - float(box_b[1]))
    return inter / (a + b - inter + EPS)


def best_match(gt_bbox, gt_cid, preds, iou_thresh=IOU_THRESH, score_thresh=0.25,
               class_aware=True) -> tuple:
    """在 preds={boxes,labels,scores} 中找与 gt 的最佳匹配。
    命中条件：类别一致 & IoU>=iou_thresh & score>=score_thresh。返回 (是否命中, 置信度)。"""
    boxes = preds.get("boxes")
    if boxes is None or len(boxes) == 0:
        return False, 0.0
    best_iou, best_score, hit = 0.0, 0.0, False
    for i in range(len(boxes)):
        if class_aware and int(preds["labels"][i]) != int(gt_cid):
            continue
        if float(preds["scores"][i]) < score_thresh:
            continue
        v = iou_xyxy(gt_bbox, boxes[i])
        if v > best_iou:
            best_iou, best_score = v, float(preds["scores"][i])
            if v >= iou_thresh:
                hit = True
    return hit, best_score


def evaluate_targets(eligible, preds_by_image) -> dict:
    """对 eligible 目标统计命中数与每目标最佳匹配置信度。"""
    n_eligible = len(eligible)
    n_correct = 0
    confs = []
    for t in eligible:
        preds = preds_by_image.get(t["image_id"],
                                   {"boxes": np.zeros((0, 4)), "labels": np.zeros((0,)),
                                    "scores": np.zeros((0,))})
        hit, score = best_match(t["bbox"], t["cid"], preds)
        if hit:
            n_correct += 1
        confs.append(score)
    return {"n_eligible": n_eligible, "n_correct": n_correct,
            "recall": n_correct / max(1, n_eligible), "confs": confs}


def hide_rate(recall: float) -> float:
    return 1.0 - recall


def attack_gain(hide_adv: float, hide_occ: float) -> float:
    """AttackGain = clip(max(0, (Hide_adv-Hide_occ)/(1-Hide_occ+eps)), 0, 1)。"""
    if hide_occ >= 1.0 - EPS:
        return 0.0 if hide_adv <= hide_occ else min(1.0, (hide_adv - hide_occ) / EPS)
    return float(min(1.0, max(0.0, (hide_adv - hide_occ) / (1.0 - hide_occ + EPS))))


def confidence_drop(conf_clean, conf_adv) -> float:
    drops = [max(0.0, (cc - ca) / (cc + EPS)) for cc, ca in zip(conf_clean, conf_adv)]
    return float(np.mean(drops)) if drops else 0.0


def attack_score_base(gain: float, conf_drop: float) -> float:
    return 0.85 * gain + 0.15 * conf_drop


def tpr_fpr_tnr(y_true, y_pred_bin) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred_bin = np.asarray(y_pred_bin).astype(int)
    tp = int(np.sum((y_pred_bin == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred_bin == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred_bin == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred_bin == 0) & (y_true == 0)))
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    tnr = tn / max(1, fp + tn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "tpr": float(tpr),
            "fpr": float(fpr), "tnr": float(tnr), "balanced_accuracy": float((tpr + tnr) / 2.0)}


def auroc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def defense_score(balanced_accuracy: float, auroc_val: float) -> float:
    return 0.7 * balanced_accuracy + 0.3 * auroc_val


def evasion_rate(tpr: float) -> float:
    return 1.0 - tpr


def attack_score_matchup(attack_base: float, evasion: float) -> float:
    return attack_base * (0.8 + 0.2 * evasion)


def final_attack_score(matchup_scores) -> float:
    return float(np.mean(matchup_scores)) if matchup_scores else 0.0


def final_defense_score(matchup_scores) -> float:
    return float(np.mean(matchup_scores)) if matchup_scores else 0.0


# --------------------------------------------------------------------------- #
# eligible 目标（干净可检测，赛题 §10）
# --------------------------------------------------------------------------- #
def compute_eligible(detector, samples):
    eligible = []
    for s in samples:
        preds = detector.predict(s["image"])[0]
        for ann in s["annotations"]:
            if int(ann["category_id"]) != TARGET_COCO_ID:
                continue
            hit, _ = best_match(ann["bbox"], TARGET_COCO_ID, preds)
            if hit:
                eligible.append({"image_id": s["image_id"], "bbox": ann["bbox"],
                                 "cid": TARGET_COCO_ID, "image": s["image"]})
    return eligible


# --------------------------------------------------------------------------- #
# 基线攻击池（A0）：random_noise / checkerboard / gray_patch + 噪声变体
# --------------------------------------------------------------------------- #
def baseline_patches(rng, patch_size=PATCH_SIZE):
    patches = []
    for bn in ["random_noise", "checkerboard", "gray_patch"]:
        a = ATTACKS[bn](seed=42)
        patches.append(a(None, {"patch_size": patch_size})["patch"])
    for _ in range(5):  # 多种随机初始化的噪声贴片（增加训练多样性）
        patches.append(np.random.default_rng(int(rng.integers(1, 10 ** 9))).random(
            (3, patch_size, patch_size), dtype=np.float32))
    return patches


def build_defense_training_data(bundle, rng, patch_size=PATCH_SIZE):
    """用基线攻击池 A0 渲染防御训练数据（clean + patched），返回 (clean_imgs, patched_imgs)。"""
    patches = baseline_patches(rng, patch_size)
    n = len(bundle["defense_train"])
    half = n // 2
    clean_imgs = [s["image"].copy() for s in bundle["defense_train"][:half]]
    patched_imgs = []
    for s in bundle["defense_train"][half:2 * half]:
        anns = s["annotations"]
        if not anns:
            continue
        ann = anns[int(rng.integers(len(anns)))]
        patch = patches[int(rng.integers(len(patches)))]
        p_img, _ = apply_patch(s["image"], ann["bbox"], patch, rng)
        patched_imgs.append(p_img)
    return clean_imgs, patched_imgs


# --------------------------------------------------------------------------- #
# 攻击评测（单攻击）
# --------------------------------------------------------------------------- #
def evaluate_attack(detector, bundle, attack_name="gradient_patch", seed=SEED,
                    iters=250, cache_dir=CACHE_DIR, save_patch_path=True) -> dict:
    env = AttackEnv(detector, device=DEVICE, patch_size=PATCH_SIZE)
    task = build_task(env, bundle)

    # 生成贴片
    t0 = time.time()
    if attack_name == "gradient_patch":
        atk = GradientPatchAttack(iters=iters, seed=seed)
    else:
        atk = ATTACKS[attack_name](seed=seed)
    result = atk(env, task)
    ok, patch, msgs = validate_patch(result)
    for m in msgs:
        print(f"[eval] 攻击 {attack_name}: {m}")
    if not ok:
        return {"name": attack_name, "valid": False, "errors": msgs}
    print(f"[eval] 攻击 {attack_name}: 贴片生成完成 ({time.time() - t0:.1f}s), "
          f"shape={patch.shape}")

    # 隐藏集 eligible + 渲染 adv/occ + 推理
    rng = np.random.default_rng(seed)
    eligible = compute_eligible(detector, bundle["hidden_test"])
    print(f"[eval] eligible 目标数: {len(eligible)}")

    preds_clean, preds_adv, preds_occ = {}, {}, {}
    imgs_by_id = {}
    for t in eligible:
        iid = t["image_id"]
        if iid not in preds_clean:
            imgs_by_id[iid] = t["image"]
            preds_clean[iid] = detector.predict(t["image"])[0]
            adv_img, params = apply_patch(imgs_by_id[iid], t["bbox"], patch, rng)
            occ_img, _ = apply_occlusion(imgs_by_id[iid], t["bbox"], params, rng)
            preds_adv[iid] = detector.predict(adv_img)[0]
            preds_occ[iid] = detector.predict(occ_img)[0]

    clean_eval = evaluate_targets(eligible, preds_clean)
    adv_eval = evaluate_targets(eligible, preds_adv)
    occ_eval = evaluate_targets(eligible, preds_occ)

    hide_adv = hide_rate(adv_eval["recall"])
    hide_occ = hide_rate(occ_eval["recall"])
    gain = attack_gain(hide_adv, hide_occ)
    conf_drop = confidence_drop(clean_eval["confs"], adv_eval["confs"])
    base = attack_score_base(gain, conf_drop)
    qualified = bool(gain >= 0.40 and hide_adv >= 0.50)

    if save_patch_path:
        save_patch(patch, os.path.join(cache_dir, "attack_patch.npy"))

    return {
        "name": attack_name, "valid": True,
        "n_eligible": len(eligible),
        "recall_clean": clean_eval["recall"],
        "recall_adv": adv_eval["recall"],
        "recall_occ": occ_eval["recall"],
        "hide_adv": hide_adv, "hide_occ": hide_occ,
        "attack_gain": gain, "conf_drop": conf_drop,
        "attack_score_base": base, "qualified": qualified,
        "patch_shape": list(patch.shape),
        "patch": patch,  # 内部字段，落盘时移除
    }


# --------------------------------------------------------------------------- #
# 防御评测（ResNet18Classifier）
# --------------------------------------------------------------------------- #
def evaluate_defense(bundle, rng, seed=SEED, epochs=8, cache_dir=CACHE_DIR) -> dict:
    # 1) 训练数据
    tr_clean, tr_patched = build_defense_training_data(bundle, rng)
    print(f"[eval] 防御训练数据: {len(tr_clean)} clean + {len(tr_patched)} patched")
    # 缓存训练数据（供 defense.py 独立运行）
    n_max = max(len(tr_clean), len(tr_patched))
    clean_arr = np.stack([im for im in tr_clean] + [tr_clean[0]] * (n_max - len(tr_clean)))
    patched_arr = np.stack([im for im in tr_patched] + [tr_patched[0]] * (n_max - len(tr_patched)))
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(os.path.join(cache_dir, "defense_train_data.npz"),
                        clean_images=clean_arr, patched_images=patched_arr)

    # 2) 训练 ResNet18Classifier
    t0 = time.time()
    defense = ResNet18Classifier(epochs=epochs, seed=seed)
    model = defense.fit(None, tr_clean, tr_patched)
    print(f"[eval] ResNet18Classifier 训练完成 ({time.time() - t0:.1f}s)")

    # 3) 验证集：clean（FPR）+ 基线池 patched（TPR_baseline）-> 资格线
    patches = baseline_patches(rng)
    val_clean = [s["image"].copy() for s in bundle["defense_val"]]
    val_patched = []
    for s in bundle["defense_val"]:
        anns = s["annotations"]
        if not anns:
            val_patched.append(s["image"].copy())
            continue
        ann = anns[int(rng.integers(len(anns)))]
        patch = patches[int(rng.integers(len(patches)))]
        p_img, _ = apply_patch(s["image"], ann["bbox"], patch, rng)
        val_patched.append(p_img)
    y_true = np.concatenate([np.zeros(len(val_clean)), np.ones(len(val_patched))])
    scores = model.predict_many(val_clean + val_patched)
    tf = tpr_fpr_tnr(y_true, (scores >= 0.5).astype(int))
    au = auroc(y_true, scores)
    dscore = defense_score(tf["balanced_accuracy"], au)
    qualified = bool(tf["fpr"] <= 0.10 and tf["tpr"] >= 0.70)
    print(f"[eval] 验证集(基线池): TPR={tf['tpr']:.3f} FPR={tf['fpr']:.3f} "
          f"BalAcc={tf['balanced_accuracy']:.3f} AUROC={au:.3f} "
          f"DefenseScore={dscore:.3f} qualified={qualified}")

    return {
        "name": "resnet18_classifier", "trained": True,
        "n_train_clean": len(tr_clean), "n_train_patched": len(tr_patched),
        "epochs": epochs,
        "tpr_baseline": tf["tpr"], "fpr": tf["fpr"], "tnr": tf["tnr"],
        "balanced_accuracy": tf["balanced_accuracy"], "auroc": au,
        "defense_score": dscore, "qualified": qualified,
        "model": model,  # 内部字段，落盘时移除
    }


# --------------------------------------------------------------------------- #
# 攻击 × 防御对阵（隐藏阶段未知贴片）
# --------------------------------------------------------------------------- #
def evaluate_matchup(patch, model, bundle, rng) -> dict:
    val_clean = [s["image"].copy() for s in bundle["hidden_test"]]
    val_patched = []
    for s in bundle["hidden_test"]:
        anns = s["annotations"]
        if not anns:
            val_patched.append(s["image"].copy())
            continue
        ann = anns[int(rng.integers(len(anns)))]
        p_img, _ = apply_patch(s["image"], ann["bbox"], patch, rng)
        val_patched.append(p_img)
    y_true = np.concatenate([np.zeros(len(val_clean)), np.ones(len(val_patched))])
    scores = model.predict_many(val_clean + val_patched)
    tf = tpr_fpr_tnr(y_true, (scores >= 0.5).astype(int))
    au = auroc(y_true, scores)
    ev = evasion_rate(tf["tpr"])
    return {"tpr": tf["tpr"], "fpr": tf["fpr"], "tnr": tf["tnr"],
            "balanced_accuracy": tf["balanced_accuracy"], "auroc": au,
            "evasion": ev, "defense_score": defense_score(tf["balanced_accuracy"], au)}


# --------------------------------------------------------------------------- #
# 评测入口
# --------------------------------------------------------------------------- #
def evaluate(stage="attack", seed=SEED, iters=250, epochs=8,
             cache_dir=CACHE_DIR, force_detector=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    bundle = load_benchmark(cache_dir=cache_dir)
    # 统一检测模型
    detector = build_detector(cache_dir=cache_dir, force_train=force_detector)
    print(f"[eval] 检测器: {detector.name} (score_thresh={detector.score_thresh}, "
          f"nms_iou={detector.nms_iou}) 数据加载+检测器就绪 ({time.time() - t0:.1f}s)")

    result = {
        "stage": stage,
        "seed": seed,
        "dataset": "synthetic_person_detection",
        "image_size": 128, "patch_size": PATCH_SIZE,
        "detector": detector.name,
        "det_score_thresh": detector.score_thresh,
        "det_iou_thresh": IOU_THRESH,
    }

    # ---- 攻击评测 ----
    rng_attack = np.random.default_rng(seed)
    atk_res = evaluate_attack(detector, bundle, attack_name="gradient_patch",
                              seed=seed, iters=iters, cache_dir=cache_dir)
    patch = atk_res.pop("patch", None)
    for k, v in atk_res.items():
        if k != "name":
            result[f"attack_{k}"] = v
    result["attack_name"] = "gradient_patch"

    # ---- 防御评测 ----
    if stage in ("defense", "all"):
        rng_def = np.random.default_rng(seed + 7)
        def_res = evaluate_defense(bundle, rng_def, seed=seed, epochs=epochs,
                                   cache_dir=cache_dir)
        model = def_res.pop("model", None)
        for k, v in def_res.items():
            if k != "name":
                result[f"defense_{k}"] = v
        result["defense_name"] = "resnet18_classifier"

        # 隐藏阶段"未知贴片"对阵（攻击方实际提交的梯度贴片，防御训练时未见过）
        if patch is not None and model is not None:
            mu = evaluate_matchup(patch, model, bundle,
                                  np.random.default_rng(seed + 1))
            result["unseen_tpr"] = mu["tpr"]
            result["unseen_fpr"] = mu["fpr"]
            result["unseen_balanced_accuracy"] = mu["balanced_accuracy"]
            result["unseen_auroc"] = mu["auroc"]
            result["unseen_evasion"] = mu["evasion"]
            result["unseen_defense_score"] = mu["defense_score"]
            print(f"[eval] 未知贴片对阵: TPR={mu['tpr']:.3f} FPR={mu['fpr']:.3f} "
                  f"BalAcc={mu['balanced_accuracy']:.3f} AUROC={mu['auroc']:.3f} "
                  f"Evasion={mu['evasion']:.3f} DefenseScore={mu['defense_score']:.3f}")

    # ---- 全对阵矩阵与最终得分 ----
    if stage == "all" and model is not None:
        matrix = {}
        rng_m = np.random.default_rng(seed + 2)
        for an in ["random_noise", "checkerboard", "gray_patch", "gradient_patch"]:
            if an == "gradient_patch":
                p = patch
            else:
                a = ATTACKS[an](seed=seed)
                p = a(None, {"patch_size": PATCH_SIZE})["patch"]
            mu = evaluate_matchup(p, model, bundle, rng_m)
            matrix[an] = mu
            print(f"[eval] 阵 {an} x resnet18_classifier: Evasion={mu['evasion']:.3f} "
                  f"AttackScore={attack_score_matchup(atk_res['attack_score_base'] if an == 'gradient_patch' else 0.0, mu['evasion']):.3f} "
                  f"DefenseScore={mu['defense_score']:.3f}")
        # 最终得分（本基准仅交付一个防御方法 D=resnet18_classifier；攻击方得分 =
        # 交付基线面对全部防御方法的 AttackScore 均值，即其单对阵得分；
        # 防御方得分 = 面对全部攻击方法的 DefenseScore 均值）
        atk_base = atk_res["attack_score_base"]
        result["matrix"] = {k: {kk: vv for kk, vv in v.items()} for k, v in matrix.items()}
        result["final_attack_score"] = final_attack_score(
            [attack_score_matchup(atk_base, matrix["gradient_patch"]["evasion"])])
        result["final_defense_score"] = final_defense_score(
            [matrix[an]["defense_score"] for an in matrix])
    return result


# --------------------------------------------------------------------------- #
# 报告输出
# --------------------------------------------------------------------------- #
def print_report(r: dict):
    print("\n" + "=" * 66)
    print(f"基准: 目标检测对抗贴片  |  阶段: {r['stage']}  |  种子: {r['seed']}")
    print("=" * 66)
    print(f"数据集: {r['dataset']}  检测器: {r['detector']} (τ_det={r['det_score_thresh']})")
    if "attack_name" in r:
        print("-" * 66)
        print(f"攻击方: {r['attack_name']}")
        print(f"  eligible 目标数    : {r['attack_n_eligible']}")
        print(f"  Recall_clean       = {r['attack_recall_clean']:.4f}")
        print(f"  Recall_adv         = {r['attack_recall_adv']:.4f}   HideRate_adv = {r['attack_hide_adv']:.4f}")
        print(f"  Recall_occ         = {r['attack_recall_occ']:.4f}   HideRate_occ = {r['attack_hide_occ']:.4f}")
        print(f"  AttackGain         = {r['attack_attack_gain']:.4f}  (资格线 ≥ 0.40)")
        print(f"  ConfidenceDrop     = {r['attack_conf_drop']:.4f}")
        print(f"  AttackScore_base   = {r['attack_attack_score_base']:.4f}")
        print(f"  攻击资格            : AttackGain≥0.40 且 HideRate_adv≥0.50 -> {r['attack_qualified']}")
    if "defense_name" in r:
        print("-" * 66)
        print(f"防守方: {r['defense_name']}（训练 {r['defense_n_train_clean']} clean + "
              f"{r['defense_n_train_patched']} patched, {r['defense_epochs']} epochs）")
        print(f"  验证集 TPR_baseline = {r['defense_tpr_baseline']:.4f}")
        print(f"  验证集 FPR          = {r['defense_fpr']:.4f}  (资格线 ≤ 0.10)")
        print(f"  BalancedAccuracy    = {r['defense_balanced_accuracy']:.4f}")
        print(f"  AUROC               = {r['defense_auroc']:.4f}")
        print(f"  DefenseScore        = {r['defense_defense_score']:.4f}")
        print(f"  防御资格             : FPR≤0.10 且 TPR≥0.70 -> {r['defense_qualified']}")
        print(f"  未知贴片对阵 (gradient_patch):")
        print(f"    TPR={r['unseen_tpr']:.4f}  FPR={r['unseen_fpr']:.4f}  "
              f"BalAcc={r['unseen_balanced_accuracy']:.4f}  AUROC={r['unseen_auroc']:.4f}  "
              f"Evasion={r['unseen_evasion']:.4f}")
        print(f"    DefenseScore_unknown = {r['unseen_defense_score']:.4f}")
    if "final_attack_score" in r:
        print("-" * 66)
        print(f"全对阵: FinalAttackScore = {r['final_attack_score']:.4f}  "
              f"FinalDefenseScore = {r['final_defense_score']:.4f}")
    print("=" * 66)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="目标检测对抗贴片基准：Baseline 指标计算",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=["attack", "defense", "all"],
                        default="attack", help="评测阶段")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--iters", type=int, default=250, help="梯度贴片优化迭代步数")
    parser.add_argument("--epochs", type=int, default=8, help="ResNet18 训练轮数")
    parser.add_argument("--force-detector", action="store_true", help="强制重训检测器")
    parser.add_argument("--output", default="results/metrics.json")
    args = parser.parse_args()

    res = evaluate(stage=args.stage, seed=args.seed, iters=args.iters,
                   epochs=args.epochs, force_detector=args.force_detector)
    print_report(res)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n[test] 指标已保存: {args.output}")
