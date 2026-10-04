# 精密kスイープ延長ラン(k=99, 64, 80)をローカルPCで実行する

クラウド側のコンテナは、セッションを開いていない間に落ちると誰も再開できず、
直近では3日間ランが完全に停止していた。チャンク・チェックポイント(500世代ごと、
中断なし実行とビット一致することをテスト済み)があるので、以後はユーザー自身の
PCで `nohup` 実行に切り替える。このファイルはそのための手順書。

対象ブランチ: `claude/lindgren-population-evolution-08fat1`
対象リポジトリ: `dpforest610/EvoPersonality_ABM`

---

## 0. 引き継ぎ時点の進捗

クラウド側のランはこの手順書を作る前に停止済み。GitHubには以下がpush済み。

- 完了済み結果(最終CSV): メインブランチに通常コミットで入っている
- 進行中のチェックポイント(.pkl): `checkpoint-scratch` ブランチに、履歴を持たない1件だけのコミットとして入っている(下の「1-3」でこれを取り出す)

| k | 完了シード数 | 進行中チェックポイント(seed: 到達世代/5000) |
|---|---|---|
| 99 | 16/50 | seed16: 1000, seed17: 500, seed18: 500, seed19: 500 |
| 64 | 0/50 | なし |
| 80 | 0/50 | なし |

(合計 150ユニット中16完了。進行中の4シードも、保存済みの世代まではそのまま使われる。)

---

## 1. 準備

### 1-1. Python と依存パッケージ

**macOS**

```bash
# Python 3.11系を推奨(3.10以上で動くはず)。Homebrewがなければ https://brew.sh を先に。
brew install python@3.11 git
python3.11 --version
```

**Windows (WSL2 Ubuntu)**

WSL2がまだなら、PowerShell(管理者)で:

```powershell
wsl --install
```

再起動後、Ubuntuのターミナルを開いて:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
python3 --version
```

以降のコマンドはすべて、macOSのターミナル、またはWSL2 Ubuntuのターミナル(Windows単体のPowerShell/コマンドプロンプトではない)で実行する。

### 1-2. clone と作業ブランチへの切り替え

```bash
git clone https://github.com/dpforest610/EvoPersonality_ABM.git
cd EvoPersonality_ABM
git checkout claude/lindgren-population-evolution-08fat1
```

(プライベートリポジトリなので、普段GitHubをcloneするときと同じ認証(SSH鍵 or Personal Access Token)が必要。)

### 1-3. 仮想環境と依存パッケージのインストール

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

(以後、新しいターミナルを開くたびに `cd EvoPersonality_ABM && source venv/bin/activate` が必要。)

### 1-4. 進行中チェックポイントの取得

`checkpoint-scratch` ブランチには、クラウド側で進行中だった4シード分のチェックポイントだけが入っている。現在のブランチ(working tree)に、そのファイルだけを取り出す:

```bash
git fetch origin checkpoint-scratch
git checkout origin/checkpoint-scratch -- scripts/output/opponent_count_precision_sweep/
git restore --staged scripts/output/opponent_count_precision_sweep/
```

(1行目・2行目はブランチを切り替えるコマンドではない。指定したパス以下のファイルだけを、今のブランチのワーキングツリーに上書き・追加する。3行目は、取り出したチェックポイントファイルが誤ってgit add/commitの対象にならないよう、ステージを外すだけでディスク上には残す — `.gitignore`で今後も自動的に除外される。)

```bash
python3 scripts/check_progress.py --k-values 99,64,80
```

と打って、上の表(k=99が16/50、4シードが進行中)と一致することを確認する。

### 1-5. テスト

```bash
python3 -m pytest -q
```

`99 passed` のように、全件 passed になることを確認する(1件でもfailしたら、2-1に進まずまず報告して)。

---

## 2. 起動

### 2-1. コア数の確認とワーカー数の決定

```bash
# macOS
sysctl -n hw.ncpu
# WSL2 Ubuntu
nproc
```

表示された数から **1引いた数** を `--workers` に指定するのを推奨(OSや他の作業に1コア残す)。例えばコア数が8なら `--workers 7`。

### 2-2. バックグラウンド起動

```bash
mkdir -p logs
nohup python3 scripts/run_opponent_count_precision_sweep_chunked.py \
  --workers 7 \
  --status-interval-seconds 3600 \
  > logs/sweep_run.log 2>&1 &
disown
```

(`--workers 7` は8コアの例。自分のコア数−1に置き換える。`--status-interval-seconds 3600` は、各kの完了数と進行中シードの到達世代だけをまとめた小さなファイル(`progress/opponent_count_precision_sweep_status.json`)を1時間ごとにメインブランチへ自動pushする機能で、既定でオンになっている。push自体に失敗してもランは止まらず、次の周期に再試行する。

クラウド実行時に使っていた、進行中チェックポイント本体(.pkl、1個あたり数百KB)を`checkpoint-scratch`ブランチへ定期pushする機能は、ローカル実行では既定でオフになっている(ローカルディスクに残っていれば十分なため)。これが必要な場合だけ `--push-checkpoints` を追加する。)

起動したら、いったんターミナルを閉じても `nohup` + `disown` によりプロセスは生き続ける(PCの電源が落ちない限り)。

### 2-3. スリープ対策

バックグラウンド実行中でも、PC自体がスリープすると処理は止まる。

**macOS**: ランを起動するときに `caffeinate` を前に付けて、スリープを防止する。

```bash
caffeinate -i nohup python3 scripts/run_opponent_count_precision_sweep_chunked.py \
  --workers 7 \
  --status-interval-seconds 3600 \
  > logs/sweep_run.log 2>&1 &
disown
```

ノートPCなら、蓋を閉じるとスリープすることに注意(`caffeinate`はディスプレイスリープ・システムスリープは防ぐが、蓋を閉じた際のスリープは別設定が必要な場合がある。「システム設定→ロック画面」や「省エネルギー」で、電源アダプタ接続時は蓋を閉じてもスリープしない設定にしておくとより安全)。

**Windows**: WSL2はWindows本体がスリープすると一緒に止まる。Windowsの電源設定でスリープを無効にする:

設定 → システム → 電源とバッテリー → 画面とスリープ → 「電源接続時にスリープする」を「なし」に変更。

(ノートPCの場合、必ずACアダプタを接続したまま実行すること。)

---

## 3. 確認

### 3-1. 進捗を一目で見る

```bash
python3 scripts/check_progress.py --k-values 99,64,80
```

出力例:

```
     k      done  in-progress (seed:next_gen/total)
    99   16/50    16:1500/5000, 17:1000/5000, 18:1000/5000, 19:1000/5000
    64    0/50    -
    80    0/50    -

Total: 16/150 (k,seed) units complete.
```

(k=4〜50の既存9点も見たい場合は `--k-values` を省略すれば12点全部が出る。)

### 3-2. プロセスが生きているか

```bash
ps aux | grep run_opponent_count_precision_sweep_chunked | grep -v grep
```

何も表示されなければ落ちている(4-2「再開」へ)。

### 3-3. ログの末尾を見る

```bash
tail -40 logs/sweep_run.log
```

---

## 4. 停止と再開

### 4-1. 安全に停止する

```bash
pkill -f run_opponent_count_precision_sweep_chunked.py
```

チャンクの切れ目(最大500世代ごと)まで待ってから終了するので、数秒〜十数分かかることがある。`ps aux`で消えたことを確認する。

(`kill -9`で強制終了しても構わない。直前のチャンク終了時点まではチェックポイントに保存されているので、進捗は実質失われない。)

### 4-2. PC再起動・スリープ・停電の後、再開する

**2-2(または2-3)と全く同じコマンドをもう一度打つだけ。**

```bash
nohup python3 scripts/run_opponent_count_precision_sweep_chunked.py \
  --workers 7 \
  --status-interval-seconds 3600 \
  > logs/sweep_run.log 2>&1 &
disown
```

起動直後のログに `N pending (M resuming from an in-progress checkpoint)` のように表示され、`M`が0より大きければチェックポイントから再開されていることが分かる。念のため、再開直後に:

```bash
python3 scripts/check_progress.py --k-values 99,64,80
```

を打って、`in-progress` の世代数が停止前より減っていない(後退していない)ことを確認する。

(前回のプロセスが異常終了した場合、ロックファイルが残っていることがあるが、死んだプロセスのロックは自動的に検出・上書きされるので、通常は何もしなくてよい。「Another run (pid ...) already holds the lock」と出て起動を拒否された場合だけ、6「トラブル時」を参照。)

---

## 5. 結果の回収

### 5-1. 完了したユニットの結果をコミット・push

実行中、5ユニット完了ごとに自動でメインブランチへコミット・pushされる(既存のクラウド版と同じ動作)。手動で今すぐpushしたい場合:

```bash
git add scripts/output/opponent_count_precision_sweep/
git commit -m "Local run: manual checkpoint of completed units"
git push
```

(完了していないシードの `.pkl` チェックポイントファイルは `.gitignore` 対象なので、これに含まれない。)

### 5-2. 全150ユニット完了後

```bash
python3 scripts/check_progress.py --k-values 99,64,80
```

で `Total: 150/150` になったら完走。そのあと、既定のステップ2解析一式を実行する:

```bash
python3 scripts/analyze_selection_gradients_by_k.py
python3 scripts/analyze_selection_gradients_step2.py
python3 scripts/analyze_selection_gradients_step2b.py
python3 scripts/analyze_fitness_peak.py
```

(これらのスクリプトは `K_VALUES` を既存の9点のまま参照しているので、全150ユニット完了を確認したら、`K_VALUES` を12点 `[4,6,8,12,16,24,32,40,50,64,80,99]` に拡張し、全12点のWilcoxon/Holm補正表・ブレークポイント回帰・第一章k=99比較・最終図を作る、という元の依頼内容は別途対応が必要 — このスクリプト変更はこのタスクの範囲外なので、完走を確認したらこちらに声をかけてほしい。)

---

## 6. トラブル時

| 症状 | 対処 |
|---|---|
| `ModuleNotFoundError: No module named 'numpy'`(など) | `source venv/bin/activate` を忘れていないか確認。してもダメなら `pip install -r requirements.txt` をやり直す |
| `python3 scripts/...` で `No such file or directory` | リポジトリのルート(`EvoPersonality_ABM/`)にいない。`cd` し直す。`pwd` と `ls`(`scripts/`フォルダが見えるか)で確認 |
| ディスク容量不足(`No space left on device`) | `df -h .` で空き容量確認。チェックポイントファイルは1シードあたり最大数MB程度なので、通常は数百MBの空きで十分。古い `scripts/output/*_smoke/` ディレクトリなど不要なものを消してもよい |
| 起動時に `Another run (pid ...) already holds the lock` と出る | `ps aux \| grep run_opponent_count_precision_sweep_chunked` で本当に動いていないか確認。動いていないのに出る場合だけ、`rm scripts/output/opponent_count_precision_sweep/.run.lock` でロックファイルを削除してから再実行 |
| `git push` が `rejected`(リモートが先に進んでいる) | `git pull --rebase` してから再度 `git push`。コンフリクトが出た場合は、`scripts/output/` 以下のCSV/pklはこちらが勝って問題ないことが多いが、判断に迷ったら触らずに相談する |
