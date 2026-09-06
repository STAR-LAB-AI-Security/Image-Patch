#!/usr/bin/env bash
# =============================================================================
# 攻击方终端脚本 —— 目标检测对抗贴片基准（Baseline：GradientPatchAttack）
#
# 在新机器上以默认参数运行：
#     pip install -r requirements.txt
#     bash run_attack.sh
#
# 输出：Recall_clean/adv/occ、HideRate、AttackGain、ConfidenceDrop、
#       AttackScore_base、攻击资格判定（保存至 results/metrics.json）
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# 指定解释器：默认 python，可用 PYTHON 环境变量覆盖
PY="${PYTHON:-python}"

echo ">>> [run_attack] 攻击方 Baseline 评测（GradientPatchAttack, iters=250, seed=42）"
"$PY" test.py --stage attack
