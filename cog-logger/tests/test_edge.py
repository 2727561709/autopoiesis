"""M02 边界条件测试。"""
import json
import queue as queue_mod

from cog_logger import Logger


def test_unicode_event(tmp_path):
    log = Logger(log_dir=tmp_path, experiment_name="中文实验", async_mode=False, console=False)
    log.info("你好，世界", tag="认知")
    log.close()
    line = log.events_path.read_text(encoding="utf-8")
    assert "你好，世界" in line


def test_non_serializable_payload(tmp_path):
    log = Logger(log_dir=tmp_path, experiment_name="t", async_mode=False, console=False)
    log.info("with tensor", data=lambda x: x)  # 不可序列化 -> repr 兜底
    log.close()
    line = log.events_path.read_text(encoding="utf-8").splitlines()[-1]
    assert json.loads(line)["event"]["data"].startswith("<function")  # repr 字符串


def test_empty_event_mapping(tmp_path):
    log = Logger(log_dir=tmp_path, experiment_name="t", async_mode=False, console=False)
    log.log({})
    log.close()  # 不抛异常


def test_queue_full_drops_and_counts(tmp_path):
    # 用一个已满的队列模拟消费者阻塞，验证丢弃计数
    log = Logger(log_dir=tmp_path, experiment_name="t", async_mode=True,
                 console=False, queue_size=2)
    full_q = queue_mod.Queue(maxsize=1)
    full_q.put(("metric", {"x": 1}))
    log._queue = full_q
    log.metric("m", 1.0)  # 队列已满 -> 丢弃
    assert log.stats.dropped == 1
    log._async = False  # 原队列已替换，跳过后台清理
    log.close()


def test_negative_and_large_metrics(tmp_path):
    log = Logger(log_dir=tmp_path, experiment_name="t", async_mode=False, console=False)
    log.metric("m", -1e9, step=0)
    log.metric("m", 1e300, step=10**18)
    log.close()
    vals = [json.loads(l)["value"]
            for l in log.metrics_path.read_text(encoding="utf-8").splitlines()]
    assert vals == [-1e9, 1e300]
