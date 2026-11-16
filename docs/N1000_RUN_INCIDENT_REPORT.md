# N=1000本番ラン中の不具合: 原因と対策

対象: `scripts/run_scent_spatial_sweep_n1000.py`、`scripts/run_n1000_pipeline_mac.sh`(2026-10-12〜13、doeRyanoMacBook-Neo.local)。
研究の結果(シミュレーションの中身・出力ファイル)には関係しない、記録・同期まわりの修正。既存の出力ファイルは変更していない。

## (a) M2 の main が結果を書き出さずに止まった

### コードで確かめたこと

1. **完了した seed の結果を書くのは main プロセスだけ。** worker は全チャンクを終えると
   checkpoint(`next_generation=5000`)を保存し、結果を `ProcessPoolExecutor` 経由で main に返す。
   timeseries/summary/individual の書き出しと checkpoint の削除は、main ループが `future.result()` を受け取ったときにだけ行われる。
2. **git 操作は全部 main ループの中で同期的に実行され、timeout が付いていなかった**
   (`subprocess.run([...git...])`、`timeout=` も `stdin` の指定もない)。
   main が git の中で止まっても、executor は別スレッドで次の seed を worker に渡し続ける。
   その結果「計算は進むのに、何も書き出されない」状態になる。
3. `--push-checkpoints` を付けると、main は600秒ごとに `seed_*_checkpoint.pkl` を **glob で全部** 集めて、
   一時 index で commit を作り、`git push --force` する。
   完了済みでまだ回収されていない seed の checkpoint も対象に入るので、main が遅れるほど push の量が増える
   (1 seed あたり、30世代×5列×1000個体のスナップショットと5万行の個体ログを含む)。
   pickle は毎回内容が変わるので、push のたびに全量がアップロードされる。
4. 5条件が同じ scratch ブランチ名 `checkpoint-scratch-scent-spatial-n1000` に `--force` で push していた。
   互いのスナップショットを上書きし合っていた(停止とは別の問題)。

### 履歴から読み取れること

- M2(`scent/`)は、12:55の起動から07:38の再起動まで、**コミットが1件もない**。
  他の条件が毎時コミットしていた進捗スナップショットも無い。
  run_metadata のコミットはロック競合で失敗していた((b)を参照)ので、M2 の main は起動後1時間以内に git の中で止まり、
  そのまま戻らなかったことになる。起動後1時間以内に走る時間ベースの git 操作は、600秒ごとの checkpoint push だけである。
- 再起動後、0 chunk の呼び出しで14 seed分がすぐ書き出された。止まっていた間に worker が終えた seed だった、という説明と合う。

どの git コマンドで止まったか(push のネットワーク停止か、それ以外か)は、リポジトリの記録だけでは決められない。
実行マシンの `logs/n1000_production_scent.log` の最後の行で確認できる。
ただ、どの場合でも「timeout のない git 呼び出しが main をふさぐ」構造が原因で、対策は同じになる。

再現(この環境、ローカルの bare リモートで、push が戻らない ssh を使用):
旧コードは 4 seed 中 0 seed を書き出したまま停止した(最初の git 呼び出しで止まる)。
新コードは、git が止まったままでも 4/4 seed を 18 秒で書き出した。

## (b) 5条件の同時 git 操作で index.lock がぶつかる

- 全条件が同じ作業ツリーで `git add` → `git commit` → `git pull --rebase` → `git push` を順序の調整なしに実行していた。
- 証拠: 12:55に4条件が同時に起動したとき、`run_metadata.json` のコミットに成功したのは M2-low だけだった。
  **M1・M1-low・M2 の `run_metadata.json` は、今もリポジトリに無い**(M2 は再起動時にコミットされた)。
- そのほかの問題: index は全プロセスで共有されている。そのため、あるプロセスが `git add` した直後に別プロセスが `git commit` すると、
  他人のファイルを自分のメッセージでコミットしうる。また、コミットメッセージが `args.movement` から作られていて、
  M2 と M2-low がどちらも "scent" と表示され、区別できなかった。

## (c) CPU の割り当て超過

`run_n1000_pipeline_mac.sh` は、条件ごとに `WORKERS` 個の worker を起動する。最大5条件が同時に走るので、
合計は `5 × WORKERS` 個の worker と 5 つの main プロセスになる。workers=2 なら 10 プロセスで、論理コア数を超えうる。
これまで警告は出していなかった。

## 対策(コード)

- `scripts/git_serial.py`(新規)
  - `git_lock()`: プロセスをまたぐ排他ロック(`.git/evopersonality-git-op.lock`、`flock`/`msvcrt`)。
    add+commit+pull+push の一連の操作をまとめて直列化する。保持していたプロセスが死ねば OS が解放するので、古いロックは残らない。
  - `run_git()`: すべての git 呼び出しに timeout(ローカル操作300秒、ネットワーク300秒)を付け、`stdin=DEVNULL` と
    `GIT_TERMINAL_PROMPT=0` を指定する(認証プロンプトで待ち続けない)。push が timeout したら再試行しない。
  - `commit_and_push()`: `git commit -- <paths>` で **自分のファイルだけ** をコミットする。
    `pull --rebase --autostash` が失敗したら `rebase --abort` して、途中状態を残さない。
  - `BackgroundGitWorker`: git の仕事を1本のスレッドで順に処理する。定期スナップショットは「まだ実行されていない同種の仕事があれば積まない」。
  - CLI `python3 scripts/git_serial.py commit-push -m MSG paths...`: シェルスクリプトから同じロックを使うためのもの。
- `scripts/run_scent_spatial_sweep_n1000.py`
  - git 操作(metadata・バッチのコミット・進捗・checkpoint スナップショット)は、すべてバックグラウンドスレッドに渡す。
    **main ループは git を待たない。**
  - checkpoint スナップショットの対象は、**実行中の seed だけ**(完了した seed の結果は回収された時点で書き出され、checkpoint は削除される)。
    実行時点で消えているファイルは除く。
  - checkpoint を一時ファイルに書いてから置き換えるようにした(バックグラウンドの読み取りが、書きかけのファイルを見ないように)。
  - scratch ブランチを条件ごとに分けた(`checkpoint-scratch-scent-spatial-n1000-<条件ディレクトリ名>`)。
  - コミットメッセージに条件ディレクトリ名(`scent_low` など)を使うようにした。
  - `--workers` が論理コア数を超えたら警告を出す。
  - 正常終了時は、残った git の仕事の完了を最大30分待つ。Ctrl-C やエラーのときは10秒だけ待つ。結果ファイルはその前に書き出し済み。
- `scripts/run_n1000_pipeline_mac.sh`
  - 起動時に `5 × WORKERS` と論理コア数(`hw.logicalcpu`)を比べ、超えていたら警告を出す。
  - 校正結果のコミットも `git_serial.py commit-push` を通すので、本番ランと同じロックで直列化される。

## テスト

- `tests/test_git_serial.py`: 自分のパスだけをコミットすること、push 先に届くこと、scratch push が共有 index に触らないこと、
  ロックの排他、バックグラウンドの順序・まとめ・例外への耐性、timeout が 124 を返すこと、消えた checkpoint を除くこと。
- `tests/test_run_scent_spatial_sweep_n1000.py`: checkpoint の置き換え書き込み、条件ごとの scratch ブランチ、コア数超過の警告。
- 2条件を同時に走らせる end-to-end テスト(ローカルの bare リモートを使用)。全コミットが届き、失敗は0件だった。
