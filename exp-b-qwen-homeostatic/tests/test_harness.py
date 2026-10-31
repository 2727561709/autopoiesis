"""实验 Harness 的单元测试（不需要模型/显卡）。
Unit tests for the experiment harness (no model / GPU).

测的是"这一轮实验能不能回答 RSI 是否真的有效"：网格是否齐全、各组算力
是否真相等、断点续跑是否会重复或遗漏、最终产物认的对不对。
These test whether a round can actually answer whether RSI pays: is the
grid complete, is compute genuinely equal, does resume skip the right
cells, is the final artefact identified correctly.
"""
import json

import pytest

from homeo_lm.harness import (append_result, completed_keys, final_ckpt,
                              grid_plan, job_dir, job_stages, read_results,
                              run_key)

START = "runs/sft-lora-v2"
TEACHER = "teacher.pt"


class TestGrid:
    def test_key_is_stable(self):
        assert run_key("D", 1) == "D-s1"

    def test_job_dir_uses_the_key(self):
        assert job_dir("runs/harness", "B", 0).name == "B-s0"

    def test_grid_covers_every_group_and_seed(self):
        plan = grid_plan("ABCDE", (0, 1, 2), START, TEACHER)
        assert len(plan["jobs"]) == 15
        assert {(j["group"], j["seed"]) for j in plan["jobs"]} == {
            (g, s) for g in "ABCDE" for s in (0, 1, 2)}

    def test_compute_is_equal_across_groups(self):
        plan = grid_plan("ABCDE", (0, 1, 2), START, TEACHER)
        assert plan["equal_compute"]["equal"] is True


class TestJobStages:
    """组的定义必须可审计：谁调数据、谁调旋钮、顺序如何。
    Group definitions must be auditable: who tunes data, who tunes
    knobs, in what order."""

    def _names(self, g):
        return [s["name"] for s in job_stages(g, job_dir("runs/harness", g, 0),
                                              START, TEACHER)]

    def test_control_group_only_fixed_training(self):
        assert self._names("A") == ["train-fixed"]

    def test_b_tunes_knobs_only(self):
        assert self._names("B") == ["l0-cycle"]

    def test_c_tunes_data_only(self):
        assert self._names("C") == ["l1-selftrain"]

    def test_d_is_data_then_knobs(self):
        """D = L1 + L0（先调数据再调旋钮，顺序写死在表里）。
        D = L1 + L0 (data first, then knobs — the order is fixed in
        the table)."""
        assert self._names("D") == ["l1-selftrain", "l0-cycle"]

    def test_e_adds_the_learning_rule(self):
        assert self._names("E") == ["l1-selftrain", "l0-cycle",
                                    "l2-diagnose"]

    def test_commands_are_subprocess_safe(self):
        """每步都必须是可执行命令列表（不允许 shell 字符串拼接）。
        Every stage must be an argv list, never a shell string."""
        for g in "ABCDE":
            for s in job_stages(g, job_dir("runs/harness", g, 0), START,
                                TEACHER):
                assert isinstance(s["cmd"], list)
                assert all(isinstance(x, str) for x in s["cmd"])


class TestResults:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "r.jsonl"
        append_result(str(p), {"key": "A-s0", "score": 0.4})
        append_result(str(p), {"key": "A-s1", "score": 0.5})
        rows = read_results(str(p))
        assert [r["score"] for r in rows] == [0.4, 0.5]

    def test_missing_file_is_empty(self, tmp_path):
        assert read_results(str(tmp_path / "nope.jsonl")) == []

    def test_bad_lines_are_skipped_not_fatal(self, tmp_path):
        p = tmp_path / "r.jsonl"
        p.write_text('{"key":"A-s0","score":0.4}\nnot json\n',
                     encoding="utf-8")
        assert len(read_results(str(p))) == 1

    def test_completed_keys(self, tmp_path):
        p = tmp_path / "r.jsonl"
        append_result(str(p), {"key": "A-s0", "score": 0.4})
        assert completed_keys(str(p)) == {"A-s0"}


class TestFinalCkpt:
    """一格跑完，到底拿什么当结果？
    After a cell finishes, what counts as its result?"""

    def test_own_adapter_wins(self, tmp_path):
        (tmp_path / "adapter").mkdir()
        (tmp_path / "adapter" / "cfg").write_text("{}", encoding="utf-8")
        assert final_ckpt(tmp_path, START) == str(tmp_path / "adapter")

    def test_adopted_trial_is_the_result(self, tmp_path):
        """L0 周期里只有**被采纳**的那一版算结果。
        In an L0 cycle only the ADOPTED version counts."""
        (tmp_path / "l0_ledger.jsonl").write_text(
            json.dumps({"verdict": "reject", "ckpt": "runs/x"}) + "\n" +
            json.dumps({"verdict": "adopt", "ckpt": "runs/adopted"}) + "\n",
            encoding="utf-8")
        assert final_ckpt(tmp_path, START) == "runs/adopted"

    def test_no_adoption_keeps_the_parent(self, tmp_path):
        """一个都没被采纳 → 原地保留 parent（拒绝该有的样子）。
        Nothing adopted -> the parent stays, which is what a rejection
        should look like."""
        (tmp_path / "l0_ledger.jsonl").write_text(
            json.dumps({"verdict": "reject", "ckpt": "runs/x"}) + "\n",
            encoding="utf-8")
        assert final_ckpt(tmp_path, START) == START

    def test_empty_dir_keeps_the_parent(self, tmp_path):
        assert final_ckpt(tmp_path, START) == START


@pytest.mark.parametrize("group", list("ABCDE"))
def test_every_group_has_at_least_one_stage(group):
    stages = job_stages(group, job_dir("runs/harness", group, 0), START,
                        TEACHER)
    assert stages, f"group {group} would run nothing"
