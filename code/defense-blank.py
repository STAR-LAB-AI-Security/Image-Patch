class ResNet18Classifier:
    """训练防御模型（学生自选算法，无需与 baseline 相同）。

    固定接口（test.py 调用契约）：
        __init__(epochs=8, batch_size=32, lr=1e-3, weight_decay=1e-4, input_size=224, seed=42)
            # 必须能接收 test.py 传入的 epochs、seed 关键字（或 **kwargs 兜底）
        fit(denv, clean_images, patched_images, cfg=None) -> 模型对象
            # 返回对象须实现 predict(image)->float 与 predict_many(images)->np.ndarray

    backend 保留 DefenseModel 包装类与 _build_resnet18 工具可用。
    """

    name = "resnet18_classifier"

    def __init__(self, epochs: int = 8, batch_size: int = 32, lr: float = 1e-3,
                 weight_decay: float = 1e-4, input_size: int = 224, seed: int = 42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.input_size = input_size
        self.seed = seed

    def fit(self, denv, clean_images, patched_images, cfg=None) -> DefenseModel:
        """训练防御模型（学生自选算法）。

        参数
        ----
        denv           : 防御环境（demo 传 None）
        clean_images   : list[HWC uint8] 干净图
        patched_images : list[HWC uint8] 含贴片图
        cfg            : 额外配置（默认 None）

        返回
        ----
        模型对象：须实现 predict(image)->float 与 predict_many(images)->np.ndarray
        """
        pass
