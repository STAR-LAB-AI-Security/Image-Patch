#!/usr/bin/env bash
# =============================================================================
# 防守方终端脚本 —— 目标检测对抗贴片基准（Baseline：ResNet18Classifier）
#
# 在新机器上以默认参数运行：
#     pip install -r requirements.txt
#     bash run_defense.sh
#
# 输出：TPR/FPR/BalancedAccuracy/AUROC/DefenseScore、资格判定，
#       以及隐藏阶段"未知贴片"对阵指标（保存至 results/metrics.json）
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# 指定解释器：默认 python，可用 PYTHON 环境变量覆盖
PY="${PYTHON:-python}"

echo ">>> [run_defense] 防守方 Baseline 评测（ResNet18Classifier, epochs=8, seed=42）"
"$PY" test.py --stage defense
