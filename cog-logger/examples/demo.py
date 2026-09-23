"""M02 最小可运行示例：python examples/demo.py"""
from cog_logger import Logger

if __name__ == "__main__":
    log = Logger(log_dir="runs", experiment_name="demo", async_mode=False)

    log.info("agent born", latent_dim=128)
    log.metric("loss", 0.42, step=1)
    log.metric("loss", 0.31, step=2)
    log.warning("energy low", energy=0.12)

    log.close()
    print("events ->", log.events_path)
    print("metrics ->", log.metrics_path)
