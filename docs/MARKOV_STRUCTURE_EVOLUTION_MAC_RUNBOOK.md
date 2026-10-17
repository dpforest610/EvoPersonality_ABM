# Markov構造進化・本番ラン(70 seed×3000世代×2条件)をMacで実行する

パイロット(10 seed×2000世代、両ノイズ条件)はクラウド環境で完走・分析済み
(`docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md` 1.4節・`scripts/output/figures/markov_sweep_pilot_summary.png`)。
本番は規模が大きく(1条件あたり約70シード×3000世代+プローブ、計算量の見積もりは
下記2節)、クラウド環境の2時間バックグラウンド上限を何十回も再開するより、
ユーザー自身のMacで`nohup`実行する方が実際的。このファイルはそのための手順書
(`docs/SCENT_MODEL_N1000_PHASE_ANIMATION.md`・`docs/LOCAL_RUN.md`と同じ形式)。

対象ブランチ: `claude/lindgren-population-evolution-08fat1`
対象リポジトリ: `dpforest610/EvoPersonality_ABM`
対象スクリプト: `scripts/run_lindgren_markov_sweep.py`

**2026-10-17改訂**: 本番計画を変更(パイロットでの切り替え頻度の結果を受けて)。
世代数は5000→**3000**(2000世代以内に十分な切り替えが観測されたため)。seed数は
50→**70**(系統間抽出[仮想参加者の群を、別々のseedから1体ずつ作る新しい抽出方式]
で、人の最大の群である69人を、別々のseedから作れるようにするため)。seed番号は
パイロット(0〜9)と重ならないよう**100〜169**を使う。

---

## 0. 本番の2条件

| 条件 | noise_rate | 出力先 |
|---|---|---|
| 主 | 0.01 | `scripts/output/markov_sweep_production_noise01` |
| 感度分析 | 0.0 | `scripts/output/markov_sweep_production_noise00` |

共通パラメータ(パイロットと同じ、design doc 1.3/2.1/2.4節で確定済み):
population_size=100, rounds_per_duel=50, point_mutation_rate=0.05,
duplication_rate=0.02, split_rate=0.02, max_memory_depth=6,
probe_interval=50, probe_repeats=30, probe_rounds=100, main_analysis_rounds=20。

**2026-10-17改訂**: seeds=70(seed番号100〜169)、generations=3000。

---

## 1. 準備

### 1-1. Python・依存パッケージ・clone

`docs/LOCAL_RUN.md` の1-1〜1-3と同じ:

```bash
brew install python@3.11 git   # Homebrewが無ければ https://brew.sh を先に
git clone https://github.com/dpforest610/EvoPersonality_ABM.git
cd EvoPersonality_ABM
git checkout claude/lindgren-population-evolution-08fat1
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

(以後、新しいターミナルを開くたびに `cd EvoPersonality_ABM && source venv/bin/activate` が必要。)

### 1-2. テスト

```bash
python3 -m pytest -q
```

全件 passed になることを確認(1件でもfailしたら、2節に進まずまず報告して)。

### 1-3. 自分のMacでの実測(推奨)

1.24秒/世代という計算量の見積もり(design doc 1.5節)はクラウド環境で測ったもので、
実際のMacでは速度が違う可能性がある。本番に入る前に、小さく実測しておく:

```bash
time python3 scripts/run_lindgren_markov_sweep.py \
  --seed-list 999 --generations 50 --noise-rate 0.01 --workers 1 \
  --out-dir /tmp/markov_sweep_timing_check
```

表示された実時間(`real`)を50で割ると、このMacでの「1世代あたりの秒数」になる
(プローブ2回分 — generation 0とgeneration 50 — のオーバーヘッドも含むので、
世代数をもっと増やすほど比率として正確になる)。終わったら削除してよい
(`/tmp`なのでgit管理外): `rm -rf /tmp/markov_sweep_timing_check`。

---

## 2. 計算量の見積もり

パイロット(10 seed×2000世代)の実測から外挿: 1シードあたり評価段(evolution)が
約62分(1.24秒/世代×3000)、プローブ段が約5分(世代数が5000→3000になった分、
プローブ回数が101→61時点に減るので、以前の見積もり[約7〜10分]から比例縮小)、
**合計1シードあたり約67分(約1.12時間)**。

70シード×2条件=140ユニット。単純直列なら約156時間(6.5日)。ワーカー数`W`
(1条件あたり)で並列化した場合、1条件の壁時計時間は約 `ceil(70/W) × 1.12時間`。
両条件を同時に(別々の`nohup`行で)走らせる前提なら、**2条件分のワーカー数の
合計がコア数を超えないこと**(3節で確認)。

例: 8コアのMacで両条件を同時に走らせる場合、`--workers 3`を2行(合計6、2コアは
OS・他作業用に残す)なら、1条件あたり `ceil(70/3)=24` チャンク×1.12時間
**≈ 27時間/条件**(2条件同時なので合計の壁時計時間もほぼ27時間)。`--workers 4`
を2行(合計8、残り無し)なら `ceil(70/4)=18` ×1.12時間 **≈ 20時間/条件**。

(1-3節の実測で、このMacでの実際の1世代あたりの秒数を確認してから、上の見積もりを
スケールし直すこと。)

---

## 3. 起動

### 3-1. コア数の確認とワーカー数の決定

```bash
sysctl -n hw.ncpu
```

表示された数から1引いた数を、**2条件のワーカー数の合計**がその範囲に収まるように
割る。例えば8コアなら、2条件とも`--workers 3`(合計6)か、`--workers 4`+`--workers 3`
(合計7)など。両条件を同時に走らせず、1条件ずつ順番に(各条件で多めのワーカー数で)
走らせたい場合は、2-2の2行を順番に(1行目の完了を待って2行目を)実行してもよい
— その場合は壁時計時間は長くなるが、1条件あたりの計算は速くなる。

### 3-2. バックグラウンド起動(条件ごとに1行)

```bash
mkdir -p logs
nohup caffeinate -dimsu python3 scripts/run_lindgren_markov_sweep.py \
  --seeds 70 --seed-offset 100 --generations 3000 --noise-rate 0.01 --workers 3 --push \
  --out-dir scripts/output/markov_sweep_production_noise01 \
  > logs/markov_sweep_noise01.log 2>&1 &
disown

nohup caffeinate -dimsu python3 scripts/run_lindgren_markov_sweep.py \
  --seeds 70 --seed-offset 100 --generations 3000 --noise-rate 0.0 --workers 3 --push \
  --out-dir scripts/output/markov_sweep_production_noise00 \
  > logs/markov_sweep_noise00.log 2>&1 &
disown
```

(`--seed-offset 100`でseed番号100〜169を使う — パイロットのseed 0〜9と重複しない。)

(`--workers 3`は8コア・2条件同時の例。3-1で決めた数に置き換える。`caffeinate -dimsu`
でディスプレイ・システムスリープを防止 — ノートPCの蓋を閉じるとそれでも止まる場合が
あるので、電源アダプタ接続時は蓋を閉じてもスリープしない設定にしておくとより安全
[システム設定→ロック画面、または省エネルギー]。`disown`で、ターミナルを閉じても
プロセスは生き続ける[Macの電源が落ちない限り]。)

**`--push`はあるが`--push-partial-progress`は付けない**: これにより、1シードが
**完全に完了した時だけ**、そのシードの全ファイルを一括でコミット・pushする
(依頼どおり「pushは完了したファイルだけ」。クラウド側のパイロット実行で使った
`--push-partial-progress`は、クラウドセッションの事情[ターン間でworking treeを
きれいに保つ必要]に特有のもので、Mac実行では不要かつ使わない)。

---

## 4. 確認

### 4-1. 進捗を一目で見る

```bash
python3 scripts/check_markov_sweep_progress.py --n-seeds 70 --seed-offset 100 --generations 3000
```

出力例:

```
                               condition      done  in-progress (seed:age/total)
scripts/output/markov_sweep_production_noise01    3/70    103:1450/3000, 104:200/3000, 105:50/3000
scripts/output/markov_sweep_production_noise00    2/70    102:900/3000

Total: 5/140 (condition, seed) units complete.
```

### 4-2. プロセスが生きているか

```bash
ps aux | grep run_lindgren_markov_sweep | grep -v grep
```

何も表示されなければ落ちている(5-2「再開」へ)。

### 4-3. ログの末尾を見る

```bash
tail -40 logs/markov_sweep_noise01.log
tail -40 logs/markov_sweep_noise00.log
```

---

## 5. 停止と再開

### 5-1. 安全に停止する

```bash
pkill -f run_lindgren_markov_sweep.py
```

すぐに終了する。**チェックポイントは50世代おき(プローブと同じタイミング)に
保存されるので、killされた時点までに失われるのは、直前のチェックポイント以降の
最大50世代分だけ**(1.24秒/世代なら約1分)。`ps aux`で消えたことを確認する。

### 5-2. Mac再起動・スリープ・停電の後、再開する

**3-2と全く同じコマンドをもう一度打つだけ。**

起動直後のログに `N pending (M resuming from an in-progress checkpoint)` と表示され、
`M`が0より大きければチェックポイントから再開されていることが分かる。念のため:

```bash
python3 scripts/check_markov_sweep_progress.py --n-seeds 70 --seed-offset 100 --generations 3000
```

で、in-progressの世代数が停止前より後退していないことを確認する。

(再開の正しさ—中断して再開した結果が、中断せず最後まで走らせた場合とビット一致
すること—は `tests/test_run_lindgren_markov_sweep.py` でテスト済み。)

---

## 6. 結果の回収

### 6-1. 完了ユニットの確認

```bash
python3 scripts/check_markov_sweep_progress.py --n-seeds 70 --seed-offset 100 --generations 3000
```

`Total: 140/140`になったら完走。

### 6-2. 手元から見て、押し漏れがないか確認

```bash
git status --short
git log --oneline -10
```

`--push`付きで実行していれば、シードが完了するたびに自動でコミット・pushされている
はず。何か残っていれば:

```bash
git add scripts/output/markov_sweep_production_noise01/ scripts/output/markov_sweep_production_noise00/
git commit -m "Markov sweep production: manual checkpoint of completed units"
git push
```

(未完了シードの`.pkl`チェックポイントと、`_census.csv.gz`/`_moves.csv.gz`は
`.gitignore`対象なので、これに含まれない — raw move dataはローカルディスクに
残る。本番データ量はパイロットの比例外挿(パイロットが約1.4GB/670MB、シード数
7倍・プローブ時点数約1.49倍でおよそ10.4倍)で、**noise_rate=0.01が約15GB、
noise_rate=0.0が約7GB、合計約22GB**になる見込み。ディスク容量に注意。)

### 6-3. その後

完走後は、`scripts/analyze_markov_sweep_pilot_network.py`・
`scripts/analyze_markov_sweep_pilot_sampling_comparison.py`と同じ枠組み
(ただしパイロット用スクリプトなので、本番の出力先・seeds=70・seed-offset=100・
generations=3000に合わせてパラメータを変えて再実行するか、本番用に複製・調整した
スクリプトを使う)で、design doc 3.3/7.2節のマルコフ網分析一式(抽出方式(A)/(B)
両方)を本番データにかける。これは別途依頼を受けてから着手する。

---

## 7. トラブル時

| 症状 | 対処 |
|---|---|
| `ModuleNotFoundError: No module named 'numpy'`(など) | `source venv/bin/activate` を忘れていないか確認 |
| `python3 scripts/...` で `No such file or directory` | リポジトリのルート(`EvoPersonality_ABM/`)にいない。`cd`し直す |
| ディスク容量不足(`No space left on device`) | `df -h .`で空き容量確認。6-2の見積もり参照。不要な古い`scripts/output/*_smoke/`等を消してもよい |
| `git push`が`rejected` | `git pull --rebase`してから再度`git push`。コンフリクトが出た場合、`scripts/output/`以下のCSVはこちらが勝って問題ないことが多いが、判断に迷ったら触らずに相談する |
| Macがスリープして止まっていた | `caffeinate`を付け忘れていないか確認。5-2の手順で再開すれば、チェックポイントからの再開で済む(最大50世代分の損失) |
| 蓋を閉じたら止まっていた | システム設定で「電源アダプタ接続時は蓋を閉じてもスリープしない」を有効にする。再開は5-2と同じ |
