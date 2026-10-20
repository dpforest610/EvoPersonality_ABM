# Markov構造進化・本番ラン(70 seed×5000世代×2条件)をMacで実行する

パイロット(10 seed×2000世代、両ノイズ条件)はクラウド環境で完走・分析済み
(`docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md` 1.4節・`scripts/output/figures/markov_sweep_pilot_summary.png`)。
本番は事前登録済み(`docs/PREREGISTRATION_markov_structure_evolution.md`)。
規模が大きく(1条件あたり約70シード×5000世代+プローブ、計算量の見積もりは下記2節)、
クラウド環境の2時間バックグラウンド上限を何十回も再開するより、ユーザー自身のMacで
`nohup`実行する方が実際的。このファイルはそのための手順書
(`docs/SCENT_MODEL_N1000_PHASE_ANIMATION.md`・`docs/LOCAL_RUN.md`と同じ形式)。

対象ブランチ: `claude/lindgren-population-evolution-08fat1`
対象リポジトリ: `dpforest610/EvoPersonality_ABM`
対象スクリプト: `scripts/run_lindgren_markov_sweep.py`

**2026-10-17改訂(2回目)**: 世代数を3000→**5000**に戻す(事前登録の設計
[`docs/PREREGISTRATION_markov_structure_evolution.md` 3節]と合わせるため —
分析自体はprobe時点 gen 1000〜5000の平均を使う)。ユーザーのMacでの実測
(1-3節)は**1世代あたり約0.9秒(50世代で43.8秒)、6コア**。この実測に基づき
2節の見積もりを更新した。**実行順も変更**: 両条件を同時に走らせず、
**主条件(ノイズ0.01)をworkers 5で単独実行し、完了してから感度分析
(ノイズ0.0)を実行する**(3節)。

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

seeds=70(seed番号100〜169)、generations=5000。分析の時間窓は
gen 1000〜5000のプローブ時点(事前登録3節)。

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

### 1-3. 自分のMacでの実測(済み — 下記の数字を使用)

1.24秒/世代という計算量の見積もり(design doc 1.5節)はクラウド環境で測ったもので、
実際のMacでは速度が違う。ユーザー実測: **50世代で43.8秒 → 1世代あたり約0.876秒
(約0.9秒)、6コア**。2節の見積もりはこの数字を使う。

(再実測したい場合:
```bash
time python3 scripts/run_lindgren_markov_sweep.py \
  --seed-list 999 --generations 50 --noise-rate 0.01 --workers 1 \
  --out-dir /tmp/markov_sweep_timing_check
```
表示された実時間(`real`)を50で割ると「1世代あたりの秒数」になる(プローブ2回分
— generation 0とgeneration 50 — のオーバーヘッドも含むので、世代数をもっと
増やすほど比率として正確になる)。終わったら削除してよい: `rm -rf /tmp/markov_sweep_timing_check`。)

---

## 2. 計算量の見積もり

1-3節の実測(0.876秒/世代、6コア)とパイロットのプローブ比率から:
1シードあたり評価段(evolution)が**約73分**(0.876秒/世代×5000)、プローブ段が
約6分(クラウド環境での5000世代・101時点の見積もり[約7〜10分、design doc 1.5節]
を、評価段の速度比0.876/1.24≈0.71でスケール)、**合計1シードあたり約79分
(約1.32時間)**。感度分析(ノイズ0.0)はプローブのノイズ条件が1通り(主条件は
2通り)で済むのでやや軽く、1シードあたり約76分(約1.27時間)。

**実行順は逐次(同時に2条件を走らせない)**: 主条件(ノイズ0.01)を
`--workers 5`(6コア−1)で単独実行し、完了してから感度分析(ノイズ0.0)を
同じく`--workers 5`で実行する。

70シード×1.32時間/seed ÷ 5 workers = `ceil(70/5)=14`チャンク×1.32時間
**≈ 18.4時間(主条件)**。感度分析は`ceil(70/5)=14`×1.27時間**≈ 17.8時間**。
**合計(逐次)≈ 36.2時間(約1.5日)**。

---

## 3. 起動

### 3-1. コア数の確認

```bash
sysctl -n hw.ncpu
```

6コアを確認。両条件は**同時に走らせず**、1条件ずつ`--workers 5`(6コア−1、OS・
他作業用に1コア残す)で逐次実行する。

### 3-2. 主条件(ノイズ0.01)をバックグラウンド起動

```bash
mkdir -p logs
nohup caffeinate -dimsu python3 scripts/run_lindgren_markov_sweep.py \
  --seeds 70 --seed-offset 100 --generations 5000 --noise-rate 0.01 --workers 5 --push \
  --out-dir scripts/output/markov_sweep_production_noise01 \
  > logs/markov_sweep_noise01.log 2>&1 &
disown
```

(`--seed-offset 100`でseed番号100〜169を使う — パイロットのseed 0〜9と重複しない。
`caffeinate -dimsu`でディスプレイ・システムスリープを防止 — ノートPCの蓋を閉じると
それでも止まる場合があるので、電源アダプタ接続時は蓋を閉じてもスリープしない設定に
しておくとより安全[システム設定→ロック画面、または省エネルギー]。`disown`で、
ターミナルを閉じてもプロセスは生き続ける[Macの電源が落ちない限り]。)

### 3-3. 主条件の完了を待ってから、感度分析(ノイズ0.0)を起動

4節の方法で主条件が`70/70`完了したことを確認してから:

```bash
nohup caffeinate -dimsu python3 scripts/run_lindgren_markov_sweep.py \
  --seeds 70 --seed-offset 100 --generations 5000 --noise-rate 0.0 --workers 5 --push \
  --out-dir scripts/output/markov_sweep_production_noise00 \
  > logs/markov_sweep_noise00.log 2>&1 &
disown
```

**`--push`はあるが`--push-partial-progress`は付けない**: これにより、1シードが
**完全に完了した時だけ**、そのシードの全ファイルを一括でコミット・pushする
(依頼どおり「pushは完了したファイルだけ」。クラウド側のパイロット実行で使った
`--push-partial-progress`は、クラウドセッションの事情[ターン間でworking treeを
きれいに保つ必要]に特有のもので、Mac実行では不要かつ使わない)。

---

## 4. 確認

### 4-1. 進捗を一目で見る

```bash
python3 scripts/check_markov_sweep_progress.py --n-seeds 70 --seed-offset 100 --generations 5000
```

出力例:

```
                               condition      done  in-progress (seed:age/total)
scripts/output/markov_sweep_production_noise01    3/70    103:2900/5000, 104:200/5000, 105:50/5000
scripts/output/markov_sweep_production_noise00    2/70    102:900/5000

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
最大50世代分だけ**(0.876秒/世代なら約44秒)。`ps aux`で消えたことを確認する。

### 5-2. Mac再起動・スリープ・停電の後、再開する

**止まっていた方の条件のコマンド(主条件が途中なら3-2、主条件は完了済みで
感度分析が途中なら3-3)を、全く同じ内容でもう一度打つだけ。**

起動直後のログに `N pending (M resuming from an in-progress checkpoint)` と表示され、
`M`が0より大きければチェックポイントから再開されていることが分かる。念のため:

```bash
python3 scripts/check_markov_sweep_progress.py --n-seeds 70 --seed-offset 100 --generations 5000
```

で、in-progressの世代数が停止前より後退していないことを確認する。

(再開の正しさ—中断して再開した結果が、中断せず最後まで走らせた場合とビット一致
すること—は `tests/test_run_lindgren_markov_sweep.py` でテスト済み。)

---

## 6. 結果の回収

### 6-1. 完了ユニットの確認

```bash
python3 scripts/check_markov_sweep_progress.py --n-seeds 70 --seed-offset 100 --generations 5000
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
7倍・プローブ時点数約2.46倍[101/41]でおよそ17.2倍)で、**noise_rate=0.01が約24GB、
noise_rate=0.0が約12GB、合計約36GB**になる見込み。ディスク容量に注意。)

### 6-3. その後

完走後は、`docs/PREREGISTRATION_markov_structure_evolution.md`の確証的分析
(主分析=系統間抽出(B)、H1〜H3のseedブートストラップ検定、分析窓gen 1000〜5000)
と副次分析一式を本番データにかける。`scripts/analyze_markov_sweep_pilot_network.py`・
`scripts/analyze_markov_sweep_pilot_sampling_comparison.py`と同じ枠組みだが、
パイロット用スクリプトなので、本番の出力先・seeds=70・seed-offset=100・
generations=5000に合わせてパラメータを変えて再実行するか、事前登録の設計(ブート
ストラップの単位・Holm補正・分析窓など)に合わせて本番用に複製・調整したスクリプト
を使う。これは別途依頼を受けてから着手する。

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

---

## 8. 確証的分析の実行(事前登録4〜6節)

**対象ブランチ: `claude/markov-production-analysis`。**
両条件(主条件ノイズ0.01・感度分析ノイズ0.0)が完走し、`claude/lindgren-population-evolution-08fat1`
がこのブランチにマージされた後は、生の対戦記録(raw `.csv.gz`、`.gitignore`対象で
コミットには含まれないが、評価ランを実行した既存の作業フォルダには既にローカルに
存在する)を含めて、**別cloneを作らず、既存の作業フォルダでブランチを切り替えるだけ**
でよい。

### 8-1. 既存の作業フォルダでブランチを切り替える

評価ラン(1〜7節)を実行した、venv・raw probe dataがある既存の作業フォルダで:

```bash
cd EvoPersonality_ABM   # 既存の作業フォルダ
git status --short       # 未コミットの変更が無いことを確認(あれば先に相談)
git fetch origin claude/markov-production-analysis
git checkout claude/markov-production-analysis
git pull origin claude/markov-production-analysis
source venv/bin/activate   # 既存のvenv(Python 3.15)をそのまま使う。新しい依存は追加していないのでpip installは不要
python3 -m pytest -q
```

全件passedになることを確認(1件でもfailしたら進まずまず報告して)。
`git checkout`はgitignore対象のraw `.csv.gz`(評価ラン1〜7節で生成済み)を消さないので、
そのまま8-3のコマンドで読める(別clone・シンボリックリンクは不要)。

### 8-3. 実行コマンド

デフォルト値はすべて事前登録(4〜5節)に合わせてあるので、追加で指定する
フラグは最小限で済む。両条件の評価データは既に`70/70`完了しているので、
**主条件→感度分析を1つのバックグラウンドジョブとして続けて実行する**(前者が
正常終了してから後者が始まる。前者が異常終了した場合は`&&`により後者は
実行されない):

```bash
mkdir -p logs
nohup caffeinate -dimsu bash -c '
venv/bin/python scripts/run_markov_confirmatory_analysis.py \
  --data-dir scripts/output/markov_sweep_production_noise01 \
  --moves-suffix noise0p01 \
  --role primary \
  --results-dir scripts/output/markov_confirmatory_production_noise01 \
&& \
venv/bin/python scripts/run_markov_confirmatory_analysis.py \
  --data-dir scripts/output/markov_sweep_production_noise00 \
  --moves-suffix noise0 \
  --role sensitivity \
  --results-dir scripts/output/markov_confirmatory_production_noise00
' > logs/markov_confirmatory_both.log 2>&1 &
disown
```

(主条件・感度分析どちらのログも同じ`logs/markov_confirmatory_both.log`に
続けて出力される。主条件の完了は、ログ中に最初に出る`all ages done in ...s`
[primaryの`--results-dir`を含む行]で判断できる。別々に起動したい場合は、
上記の`bash -c '...'`の中身を2回に分けて、8-3の旧版と同じように個別の
`nohup caffeinate -dimsu venv/bin/python ...`としてそれぞれ実行してもよい。)

(`--moves-suffix`は評価ノイズではなく**プローブ時のノイズ**を指す
ファイル名の一部 — `run_lindgren_markov_sweep.py`は常に両方のプローブノイズ
[0.0と評価ノイズ]でプローブしているが、事前登録どおり主条件は
「評価ノイズ0.01下で進化した個体をノイズ0.01でプローブ」、感度分析は
「評価ノイズ0.0下で進化した個体をノイズ0.0でプローブ」を使うため、
`markov_sweep_production_noise01/`では`noise0p01`、`markov_sweep_production_noise00/`
では`noise0`を指定する。)

### 8-4. 実行時間・メモリの見積もり

81プローブ時点(gen 1000〜5000、50世代おき)×1世代あたりの処理。ベンチマーク
(パイロット規模のseed・group_sizeで実測した計算コストと、本番group_size=56での
実測ブートストラップ1回あたりコスト[TFT約0.48ms、AllD約0.31ms、AllC約0.34ms、
Random約2.46ms、Gradual約1.09ms]を合成)から:

- 計算(CPU): 1時点あたり約70秒
- I/O(70 seed分のraw `.csv.gz`読み込み): 1時点あたり約87.5秒
- 合計 約158秒/時点 × 81時点 ÷ `--workers 5`(デフォルト) ≈ **約43分/条件**
- 両条件(主・感度分析)合計で **約1.5時間**(逐次実行時。同時に2つ走らせると
  I/O競合でこれより遅くなる可能性があるので、逐次実行を推奨)
- メモリ: 1時点あたりraw probe dataをseedごとに読んで集計後に解放するため
  **1GB未満**(ピークはプロセス数`--workers 5`×1プロセスあたり数百MB程度)

本番前に、小規模(数時点だけ)で実測してから見積もりを更新したい場合:

```bash
time venv/bin/python scripts/run_markov_confirmatory_analysis.py \
  --data-dir scripts/output/markov_sweep_production_noise01 \
  --moves-suffix noise0p01 --role primary --window-start 1000 --window-end 1100 \
  --results-dir /tmp/markov_confirmatory_timing_check
```
(`window-end 1100`で3時点[1000,1050,1100]だけ処理。終わったら
`rm -rf /tmp/markov_confirmatory_timing_check`。)

### 8-5. 進捗・再開

時点(age)ごとに`<results-dir>/_age_cache_<role>/age_<N>.pkl`へ結果をキャッシュする
(`.gitignore`対象 — pushに含まれない)。`pkill`や停電で止まっても、同じコマンドを
もう一度打てば、キャッシュ済みの時点は再計算せずスキップして再開する。

```bash
ps aux | grep run_markov_confirmatory_analysis | grep -v grep   # 生きているか確認
tail -40 logs/markov_confirmatory_both.log                       # 進捗ログ(主条件・感度分析とも)
ls scripts/output/markov_confirmatory_production_noise01/_age_cache_primary/ | wc -l   # 完了済み時点数(81で全完了)
```

(`bash -c '... && ...'`で起動しているため、再開したい場合も8-3のコマンドを
そのまま再度打てばよい — 主条件が既に全時点キャッシュ済みなら即座にスキップされ、
感度分析から実質的に再開される。)

### 8-6. 付随する図・分析スクリプト

生のraw probe dataは不要(8-3の記述的CSVだけ読む)なので、Mac以外(このクラウド
セッションなど)でも、CSVを転送すれば実行できる。各役割(primary/sensitivity)ごとに:

```bash
venv/bin/python scripts/plot_markov_confirmatory_trends.py \
  --descriptive-csv scripts/output/markov_confirmatory_production_noise01/markov_confirmatory_descriptive_primary.csv \
  --role primary --out-dir scripts/output/figures

venv/bin/python scripts/plot_markov_confirmatory_sampling_comparison.py \
  --descriptive-csv scripts/output/markov_confirmatory_production_noise01/markov_confirmatory_descriptive_primary.csv \
  --role primary --out-dir scripts/output/figures

venv/bin/python scripts/analyze_markov_confirmatory_regime_correspondence.py \
  --generation-stats-dir scripts/output/markov_sweep_production_noise01 \
  --descriptive-csv scripts/output/markov_confirmatory_production_noise01/markov_confirmatory_descriptive_primary.csv \
  --role primary --out-dir scripts/output/figures
```

(`--role sensitivity`の場合は`noise01`→`noise00`、`primary`→`sensitivity`に
読み替える。`analyze_markov_confirmatory_regime_correspondence.py`は`generation_stats.csv`
[各seedの評価段の統計、軽量で既にpush済み]を読むので、`--generation-stats-dir`は
対応する評価データのディレクトリを指定する。)

### 8-7. 結果のpush — 必ず新しいブランチへ

**`claude/markov-production-analysis`ブランチにのみpushする。
`claude/lindgren-population-evolution-08fat1`には絶対にpushしない**
(確証的分析の結果は分析用ブランチのものであり、評価ランのブランチに戻す必要はない)。

```bash
git branch --show-current   # claude/markov-production-analysis であることを確認してからpush
git status --short           # _age_cache_*/ やraw .csv.gz が含まれていないか確認
git add scripts/output/markov_confirmatory_production_noise01/markov_confirmatory_descriptive_primary.csv \
        scripts/output/markov_confirmatory_production_noise01/markov_confirmatory_results_primary.json \
        scripts/output/markov_confirmatory_production_noise00/markov_confirmatory_descriptive_sensitivity.csv \
        scripts/output/markov_confirmatory_production_noise00/markov_confirmatory_results_sensitivity.json \
        scripts/output/markov_confirmatory_regime_summary_primary.csv \
        scripts/output/markov_confirmatory_regime_summary_sensitivity.csv \
        scripts/output/figures/markov_confirmatory_*.png
git commit -m "Markov confirmatory analysis: production results (primary + sensitivity)"
git push -u origin claude/markov-production-analysis
```

(ファイル名は明示的に列挙 — `git add -A`や`git add .`は使わない。
`_age_cache_*/`は`.gitignore`で除外済みなので`git status --short`に出ないはず。
万一raw `.csv.gz`が`git status`に出たら、pushせず報告する。)
