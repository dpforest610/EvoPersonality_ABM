import importlib.util
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("git_serial", REPO_ROOT / "scripts" / "git_serial.py")
gs = importlib.util.module_from_spec(spec)
sys.modules["git_serial"] = gs
spec.loader.exec_module(gs)


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A clone of a local bare remote, with git_serial pointed at it."""
    for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    work = tmp_path / "work"
    _git(tmp_path, "clone", "-q", str(remote), str(work))
    _git(work, "checkout", "-q", "-b", "main")
    (work / "README").write_text("x\n")
    _git(work, "add", "README")
    _git(work, "commit", "-q", "-m", "init")
    _git(work, "push", "-q", "-u", "origin", "main")
    monkeypatch.setattr(gs, "REPO_ROOT", work)
    monkeypatch.setattr(gs, "LOCK_PATH", work / ".git" / "evopersonality-git-op.lock")
    return work, remote


def test_commit_and_push_commits_only_given_paths(repo):
    work, remote = repo
    (work / "mine.csv").write_text("1\n")
    (work / "theirs.csv").write_text("2\n")
    _git(work, "add", "theirs.csv")  # staged by "another process"

    assert gs.commit_and_push([Path("mine.csv")], "add mine", log=lambda m: None)

    files = _git(remote, "ls-tree", "--name-only", "main").split()
    assert "mine.csv" in files
    assert "theirs.csv" not in files
    # the other process's staged file is still staged, untouched
    assert "theirs.csv" in _git(work, "diff", "--cached", "--name-only")


def test_commit_and_push_nothing_to_commit_is_ok(repo):
    assert gs.commit_and_push([Path("README")], "no-op", log=lambda m: None)


def test_scratch_branch_push_leaves_shared_index_alone(repo):
    work, remote = repo
    ckpt = work / "seed_0_checkpoint.pkl"
    ckpt.write_bytes(pickle.dumps({"next_generation": 250}))
    assert gs.push_files_to_scratch_branch([ckpt], "snap", "scratch-x", log=lambda m: None)
    assert "seed_0_checkpoint.pkl" in _git(remote, "ls-tree", "--name-only", "scratch-x")
    assert _git(work, "status", "--porcelain", "--untracked-files=no") == ""


def test_git_lock_excludes_a_second_holder(repo):
    acquired = threading.Event()
    release = threading.Event()

    def holder():
        with gs.git_lock():
            acquired.set()
            release.wait(5)

    t = threading.Thread(target=holder)
    t.start()
    assert acquired.wait(5)
    with pytest.raises(gs.GitLockTimeout):
        with gs.git_lock(timeout=0.5):
            pass
    release.set()
    t.join()
    with gs.git_lock(timeout=2):  # free again once the holder is done
        pass


def test_background_worker_runs_in_order_coalesces_and_survives_errors():
    w = gs.BackgroundGitWorker(log=lambda m: None)
    gate = threading.Event()
    ran = []
    w.submit(lambda: gate.wait(5))  # keep the thread busy so the next jobs queue up
    w.submit(lambda: 1 / 0)  # must not kill the worker
    assert w.submit(lambda: ran.append("snap1"), key="snap")
    assert not w.submit(lambda: ran.append("snap2"), key="snap")  # coalesced: snap1 still queued
    w.submit(lambda: ran.append("batch"))
    t0 = time.time()
    gate.set()
    assert w.close(timeout=10)
    assert ran == ["snap1", "batch"]
    assert time.time() - t0 < 10


def test_run_git_reports_timeout_instead_of_hanging(repo, monkeypatch):
    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd=a[0], timeout=k.get("timeout"))
    monkeypatch.setattr(gs.subprocess, "run", fake_run)
    r = gs.run_git(["push"], timeout=1)
    assert r.returncode == 124


def test_scratch_branch_push_skips_checkpoints_deleted_since_listing(repo):
    work, remote = repo
    alive = work / "seed_1_checkpoint.pkl"
    alive.write_bytes(b"x")
    gone = work / "seed_0_checkpoint.pkl"  # seed finished; main loop already deleted it
    assert gs.push_files_to_scratch_branch([gone, alive], "snap", "scratch-y", log=lambda m: None)
    assert _git(remote, "ls-tree", "--name-only", "scratch-y").split() == ["seed_1_checkpoint.pkl"]
