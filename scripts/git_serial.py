"""Serialized, non-blocking git helpers for long unattended sweeps.

Why this exists (see the N=1000 production incident, 2026-10-12/13):
several sweep processes (one per condition) commit and push into the
SAME working copy at the same time. Without coordination they collide
on .git/index.lock (one process's `git add` fails, or another
process's `git commit` sweeps up files someone else staged), and
because every git call ran synchronously in the runner's main loop with
no timeout, one git call that never returned froze the main loop: the
workers kept finishing seeds, but nothing wrote their results.

Three pieces:

* git_lock()          -- cross-process exclusive lock (flock / msvcrt),
                         held for a whole add+commit or pull+push
                         sequence. The OS releases it if the holder dies,
                         so there is never a stale lock to clean up.
* run_git()           -- every git call gets a timeout, no stdin, and
                         GIT_TERMINAL_PROMPT=0 (a credential prompt fails
                         fast instead of waiting forever).
* BackgroundGitWorker -- one daemon thread that runs git jobs from a
                         queue, so the caller's loop never waits on git.

CLI (for shell scripts): run one command under the same lock.
    python3 scripts/git_serial.py commit-push -m "message" -- path1 path2 ...
    python3 scripts/git_serial.py run -- git status
"""

import argparse
import contextlib
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = REPO_ROOT / ".git" / "evopersonality-git-op.lock"

LOCAL_TIMEOUT = 300   # add / commit / write-tree / commit-tree
NETWORK_TIMEOUT = 300  # pull / push
LOCK_WAIT_SECONDS = 1800


class GitLockTimeout(RuntimeError):
    pass


@contextlib.contextmanager
def git_lock(timeout: float = LOCK_WAIT_SECONDS, lock_path: Optional[Path] = None):
    """Exclusive cross-process lock around a git operation sequence."""
    lock_path = LOCK_PATH if lock_path is None else lock_path
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+")
    deadline = time.monotonic() + timeout
    try:
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise GitLockTimeout(f"could not acquire {lock_path} within {timeout}s")
                    time.sleep(0.5)
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            while True:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise GitLockTimeout(f"could not acquire {lock_path} within {timeout}s")
                    time.sleep(0.5)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


def run_git(args: List[str], timeout: float = LOCAL_TIMEOUT, env: Optional[dict] = None,
            cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """git with a hard timeout; a timeout is reported as returncode 124."""
    cwd = REPO_ROOT if cwd is None else cwd
    full_env = dict(os.environ if env is None else env)
    full_env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        return subprocess.run(["git", *args], cwd=cwd, env=full_env, capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(["git", *args], 124, "", f"timed out after {timeout}s")


def commit_and_push(paths: Iterable[Path], message: str, log: Callable[[str], None] = print) -> bool:
    """add + commit ONLY `paths` (never anything another process staged),
    then pull --rebase --autostash + push, all under git_lock().
    Never raises; returns True if the commit reached the remote."""
    rel = [Path(p).as_posix() for p in paths]
    if not rel:
        return True
    try:
        with git_lock():
            add = run_git(["add", "--", *rel])
            if add.returncode != 0:
                log(f"git add failed for {message}: {add.stderr.strip()}")
                return False
            commit = run_git(["commit", "-m", message, "--", *rel])
            if commit.returncode != 0:
                if "nothing to commit" in (commit.stdout + commit.stderr) or "no changes added" in commit.stdout:
                    return True
                log(f"git commit failed for {message}: {commit.stderr.strip()}")
                return False

            delay = 2
            for attempt in range(4):
                pull = run_git(["pull", "--rebase", "--autostash"], timeout=NETWORK_TIMEOUT)
                if pull.returncode != 0:
                    log(f"git pull --rebase failed (attempt {attempt + 1}) for {message}: {pull.stderr.strip()}")
                    abort = run_git(["rebase", "--abort"])  # never leave a half-done rebase behind
                    if abort.returncode == 0:
                        log("aborted the in-progress rebase")
                push = run_git(["push"], timeout=NETWORK_TIMEOUT)
                if push.returncode == 0:
                    log(f"pushed: {message}")
                    return True
                log(f"git push failed (attempt {attempt + 1}) for {message}: {push.stderr.strip()}")
                if push.returncode == 124:
                    break  # the network is hanging, not flaky: don't stack more timeouts
                if attempt < 3:
                    time.sleep(delay)
                    delay *= 2
            log(f"git push ultimately failed for: {message} (committed locally; the next push carries it)")
            return False
    except GitLockTimeout as e:
        log(f"skipped commit/push for {message}: {e}")
        return False


def push_files_to_scratch_branch(paths: Iterable[Path], message: str, branch: str,
                                 log: Callable[[str], None] = print) -> bool:
    """Force-push a commit containing just `paths` to `branch`, built with a
    private index (the shared .git/index is never touched). Under git_lock()
    anyway, so it never runs concurrently with another process's push."""
    import tempfile
    # the list was built earlier on another thread; a seed that finished in
    # between has had its checkpoint deleted, so keep only what still exists
    rel = [Path(p).resolve().relative_to(REPO_ROOT.resolve()).as_posix() for p in paths if Path(p).exists()]
    if not rel:
        return True
    try:
        with git_lock(), tempfile.TemporaryDirectory() as tmpdir:
            env = dict(os.environ)
            env["GIT_INDEX_FILE"] = os.path.join(tmpdir, "index")
            add = run_git(["add", "-f", "--", *rel], env=env)
            if add.returncode != 0:
                log(f"scratch-branch git add failed for {message}: {add.stderr.strip()}")
                return False
            tree = run_git(["write-tree"], env=env)
            if tree.returncode != 0:
                log(f"scratch-branch write-tree failed for {message}: {tree.stderr.strip()}")
                return False
            commit = run_git(["commit-tree", tree.stdout.strip(), "-m", message], env=env)
            if commit.returncode != 0:
                log(f"scratch-branch commit-tree failed for {message}: {commit.stderr.strip()}")
                return False
            delay = 2
            for attempt in range(4):
                push = run_git(["push", "--force", "origin", f"{commit.stdout.strip()}:refs/heads/{branch}"],
                               timeout=NETWORK_TIMEOUT)
                if push.returncode == 0:
                    log(f"pushed checkpoint snapshot to {branch}: {message}")
                    return True
                log(f"scratch-branch push failed (attempt {attempt + 1}) for {message}: {push.stderr.strip()}")
                if push.returncode == 124:
                    break
                if attempt < 3:
                    time.sleep(delay)
                    delay *= 2
            log(f"scratch-branch push ultimately failed for: {message}")
            return False
    except GitLockTimeout as e:
        log(f"skipped scratch-branch push for {message}: {e}")
        return False


class BackgroundGitWorker:
    """Runs git jobs one at a time on a daemon thread.

    submit(fn, key=...) with a key coalesces: if a job with the same key is
    still queued (not yet started), the new submission is dropped -- used
    for periodic snapshots, where only the latest one matters."""

    def __init__(self, log: Callable[[str], None] = print):
        self._q: "queue.Queue" = queue.Queue()
        self._pending_keys = set()
        self._lock = threading.Lock()
        self._log = log
        self._thread = threading.Thread(target=self._run, name="git-worker", daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[[], object], key: Optional[str] = None) -> bool:
        with self._lock:
            if key is not None:
                if key in self._pending_keys:
                    return False
                self._pending_keys.add(key)
        self._q.put((fn, key))
        return True

    def busy_with(self, key: str) -> bool:
        with self._lock:
            return key in self._pending_keys

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            fn, key = item
            try:
                fn()
            except Exception as e:  # a git job must never kill the worker thread
                self._log(f"background git job failed: {e!r}")
            finally:
                if key is not None:
                    with self._lock:
                        self._pending_keys.discard(key)
                self._q.task_done()

    def close(self, timeout: Optional[float] = None) -> bool:
        """Finish all queued jobs, then stop. Returns False if `timeout` hit."""
        self._q.put(None)
        self._thread.join(timeout)
        return not self._thread.is_alive()


def _main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    cp = sub.add_parser("commit-push")
    cp.add_argument("-m", "--message", required=True)
    cp.add_argument("paths", nargs="+")
    rp = sub.add_parser("run")
    rp.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.cmd == "commit-push":
        ok = commit_and_push([Path(x) for x in args.paths], args.message,
                             log=lambda s: print(s, flush=True))
        sys.exit(0 if ok else 1)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    with git_lock():
        sys.exit(subprocess.run(command, cwd=REPO_ROOT, stdin=subprocess.DEVNULL).returncode)


if __name__ == "__main__":
    _main()
