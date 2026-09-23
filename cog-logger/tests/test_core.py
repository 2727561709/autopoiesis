"""M02 核心功能测试：写入、读取、格式、异步。"""
import json

from cog_logger import Logger


def read_jsonl(path):
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(l) for l in lines]


def test_write_and_read_events(tmp_path):
    log = Logger(log_dir=tmp_path, experiment_name="t1", async_mode=False, console=False)
    log.info("hello", step=1)
    log.warning("careful")
    log.close()
    events = read_jsonl(log.events_path)
    assert len(events) == 3  # started + hello + careful
    assert events[0]["event"]["msg"] == "logger started"
    assert events[2]["level"] == "WARNING"
    assert events[1]["event"]["step"] == 1


def test_write_and_read_metrics(tmp_path):
    log = Logger(log_dir=tmp_path, experiment_name="t2", async_mode=False, console=False)
    log.metric("loss", 0.5, step=1)
    log.metric("loss", 0.25, step=2)
    log.close()
    metrics = read_jsonl(log.metrics_path)
    assert [m["value"] for m in metrics] == [0.5, 0.25]
    assert [m["step"] for m in metrics] == [1, 2]


def test_record_format(tmp_path):
    log = Logger(log_dir=tmp_path, experiment_name="t3", async_mode=False, console=False)
    log.metric("reward", 1.0, step=5)
    log.close()
    rec = read_jsonl(log.metrics_path)[0]
    assert set(rec) >= {"ts", "exp", "name", "value", "step"}
    assert rec["exp"] == "t3"
    assert isinstance(rec["ts"], float)


def test_async_write_flush(tmp_path):
    log = Logger(log_dir=tmp_path, experiment_name="t4", async_mode=True, console=False)
    for i in range(100):
        log.metric("m", i, step=i)
    log.flush()
    assert log.stats.metrics_written == 100
    log.close()
    assert len(read_jsonl(log.metrics_path)) == 100


def test_stats(tmp_path):
    log = Logger(log_dir=tmp_path, experiment_name="t5", async_mode=False, console=False)
    log.info("a")
    log.metric("b", 1.0)
    assert log.stats.events_written == 2  # started + a
    assert log.stats.metrics_written == 1
    log.close()


def test_close_idempotent(tmp_path):
    log = Logger(log_dir=tmp_path, experiment_name="t6", async_mode=True, console=False)
    log.close()
    log.close()  # 幂等
