"""M01 最小可运行示例：python examples/demo.py"""
from cog_config import Config

if __name__ == "__main__":
    # 1. 纯默认值
    cfg = Config()
    print("defaults:", cfg.latent_dim, cfg.device)

    # 2. 从字典式关键字覆盖
    cfg = Config(latent_dim=64, experiment_name="demo")
    print("overridden:", cfg.latent_dim, cfg.experiment_name)

    # 3. 校验与导出
    cfg.validate()
    print("to_dict keys:", sorted(cfg.to_dict())[:5], "...")

    # 4. 写出 / 读回
    cfg.save("/tmp/cog_demo_config.yaml")
    cfg2 = Config.load("/tmp/cog_demo_config.yaml")
    assert cfg2 == cfg
    print("roundtrip OK")
