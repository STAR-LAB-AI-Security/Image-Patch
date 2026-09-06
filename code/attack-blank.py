class GradientPatchAttack:
    """对抗贴片攻击（学生自选算法，无需与 baseline 相同）。

    固定接口（test.py 调用契约）：
        __init__(iters=250, lr=0.01, batch_size=8, tv_weight=0.01, init="noise", seed=42)
            # 必须能接收 test.py 传入的 iters、seed 关键字（或 **kwargs 兜底）
        __call__(env, task) -> {"patch": np.ndarray (3,32,32) float32, 取值 [0,1]}

    env 提供：env.detector（predict / loss / hiding_loss）、env.sample_params(bbox, rng)、
    env.render_patch_torch(img_t, patch, params)；init_patch / _total_variation 可用。
    """

    name = "gradient_patch"

    def __init__(self, iters: int = 250, lr: float = 0.01, batch_size: int = 8,
                 tv_weight: float = 0.01, init: str = "noise", seed: int = 42):
        self.iters = iters
        self.lr = lr
        self.batch_size = batch_size
        self.tv_weight = tv_weight
        self.init = init
        self.seed = seed

    def __call__(self, env: AttackEnv, task: dict) -> dict:
        """对抗贴片攻击（学生自选算法）。

        参数
        ----
        env  : AttackEnv（env.detector / env.sample_params / env.render_patch_torch）
        task : {"images", "annotations", "target_class", "patch_size", "detector"}

        返回
        ----
        {"patch": np.ndarray (3,32,32) float32，取值 [0,1]}
        """
        pass
