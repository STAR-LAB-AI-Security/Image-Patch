# 目标检测对抗贴片 —— 代码说明（README）

合成目标检测场景上的**对抗贴片攻防**基准（轻量实现，忠实赛题概念）：攻击方生成一张受尺寸约束的**通用对抗贴片**（$(3,32,32)$），由裁判端统一渲染到目标框内，使统一检测模型对该目标漏检 / 置信度下降；防御方训练一个独立的**贴片检测二分类器**（0 = 正常图片，1 = 含贴片图片），在不知道攻击贴片内容的情况下识别含贴片图片。指标为 **Recall/HideRate、AttackGain、ConfidenceDrop、BalancedAccuracy、AUROC、DefenseScore** 等。

> 原赛题（`目标检测对抗贴片攻防对抗赛`）使用 MS COCO 2017 + 真实 YOLO/Faster R-CNN 检测器；本交付物在无 GPU 的环境下将其替换为**离线确定性合成场景**（128×128，person 类矩形目标）+ **轻量单阶段检测器 TinyDetector**（网格 16×16、3×3 检测头），概念、接口与指标公式与赛题严格一致，CPU 上分钟级跑通。

## 文件职责

| 文件 | 职责 |
|---|---|
| `dataset.py` | 生成/缓存合成目标检测场景（`data/benchmark.npz`，seed 42 确定性）；训练/缓存统一检测模型 TinyDetector（`data/detector.pt`）；裁判端贴片渲染协议（`apply_patch` / `apply_occlusion`） |
| `attack.py` | 攻击 Baseline：梯度优化通用对抗贴片（GradientPatchAttack，DPatch/EOT 思想），`attack(env, task) -> {"patch"}`；含 A0 基线池（random_noise / checkerboard / gray_patch） |
| `defense.py` | 防御 Baseline：ResNet18Classifier（ImageNet 预训练 ResNet18 二分类器，材料 D0-1），`fit(...) -> DefenseModel.predict(image) -> float` |
| `test.py` | 评测入口：生成贴片 → 渲染 → 统一检测器推理 → 全部指标；`--stage attack\|defense\|all` |
| `run_attack.sh` / `run_defense.sh` | 终端脚本：`python test.py --stage attack / --stage defense` |
| `requirements.txt` | 最小依赖：numpy / torch / torchvision / Pillow / scikit-learn |

## 运行方式

```bash
pip install -r requirements.txt
bash run_attack.sh      # 攻击方 Baseline 评测（首次运行会生成数据 + 训练检测器，约 1-3 分钟）
bash run_defense.sh     # 防守方 Baseline 评测（复用数据/检测器缓存，约 1-2 分钟）
```

也可直接运行完整全对阵评测：

```bash
python test.py --stage all
```

- 数据集与检测器均**离线**（合成数据内置生成、检测器本地训练缓存），无下载；
- 防守方 **ResNet18Classifier 首次运行需下载 ImageNet 预训练权重（约 45MB，来自 PyTorch CDN，缓存于 `~/.cache/torch`）**；下载失败时自动回退随机初始化并给出警告（性能下降）；
- 固定随机种子（默认 42）与固定数据划分，结果可复现；`data/` 下数据集、检测器权重与贴片均带缓存，重复运行秒级完成；如需强制重训：

```bash
python test.py --stage attack --force-detector     # 强制重训检测器
python attack.py --iters 400                        # 调整贴片优化步数（默认 250）
```

## 关键参数默认值

| 项 | 值 |
|---|---|
| 图像 / 贴片尺寸 | 128×128 / $(3,32,32)$ float [0,1] |
| 数据划分 | attack_train 120 / attack_val 30 / defense_train 160 / defense_val 60 / hidden_test 80 |
| 检测器 | TinyDetector：网格 16×16、stride 8、通道 16–32、3×3 检测头，12 epochs，$\tau_{det}=0.25$，NMS IoU 0.45 |
| 贴片渲染 | 面积 ≤ 目标框 80%，旋转 ±15°，缩放 [0.85,1.15]，中心限框中央 60%，颜色抖动 ±10% |
| 攻击优化 | GradientPatchAttack：iters=250、lr=0.01、batch=8、TV 正则 0.01、框回归破坏权重 2.0 |
| 防御训练 | ResNet18Classifier：输入 224×224、ImageNet 归一化、fc→1 logit、BCEWithLogits、Adam lr=1e-3、8 epochs |
| 资格线 | 攻击：AttackGain ≥ 0.40 且 HideRate_adv ≥ 0.50；防御：FPR ≤ 0.10 且 TPR ≥ 0.70 |

## 指标

$$Recall = \frac{N_{\text{correct}}}{N_{\text{eligible}}},\quad HideRate_{adv} = 1 - Recall_{adv}$$

$$AttackGain = \operatorname{clip}\left(\max\left(0,\ \frac{HideRate_{adv} - HideRate_{occ}}{1 - HideRate_{occ} + \varepsilon}\right),\ 0,\ 1\right)$$

$$BalancedAccuracy = \frac{TPR + TNR}{2},\quad DefenseScore = 0.7 \times BalancedAccuracy + 0.3 \times AUROC$$

攻击方最终得分 = 面对全部防御方法的 $AttackScore_{i,j}$ 均值（越大越优）；防御方最终得分 = 面对全部攻击方法的 $DefenseScore$ 均值（越大越优）。隐藏阶段评测使用攻击方实际提交的**未知贴片**（防御训练时未见过），用于检验未知贴片泛化检测能力。

## 实测结果（参照基线，数值随硬件/运行略有波动）

以默认参数（seed 42）在纯 CPU 上实测（154 个 eligible 目标）：

- **攻击方**（GradientPatchAttack）：$Recall_{clean}=1.0000$ → $Recall_{adv}=0.4870$（$HideRate_{adv}=0.5130$，检测性能显著下降），自然遮挡 $HideRate_{occ}=0.4286$，$AttackGain=0.1477$（对抗贴片比普通遮挡额外隐藏约 15% 的目标；合成轻量检测器对贴片较鲁棒，AttackGain 未达真实赛事的 0.40 资格线，与材料后端在合成检测器上的实测一致）；
- **防守方**（ResNet18Classifier）：验证集 $TPR=1.0000$、$FPR=0.0000$、$AUROC=1.0000$，**通过资格线**；对未知梯度贴片 $TPR_{unknown}=1.0000$、$FPR=0.0250$、$AUROC=0.9997$、$DefenseScore=0.9912$；
- **全对阵**：$FinalAttackScore = 0.1107$，$FinalDefenseScore = 0.9837$。

完整数值以运行输出 `results/metrics.json` 为准。
