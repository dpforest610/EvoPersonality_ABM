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

---

## 9. Gradualルール修正後の再プローブ(`docs/DEVIATION_gradual_rule.md`)

**対象ブランチ:`claude/markov-production-analysis`(変更なし)。**
`evopersonality_abm/strategies.py`のGradual・GradualAppeasementの罰の数え方を
人の実験データに合わせて修正した(`docs/DEVIATION_gradual_rule.md`)。**進化(evolution)は
やり直さない** — Gradual/GradualAppeasementは進化の適応度計算に使われておらず、固定
オポネントとしてのプローブにのみ使われるため、Macにある既存のgenotype census
(`seed_X_probe_age_G_census.csv.gz`、8節までの評価ランで既に生成済み)を再利用し、
**この2オポネントに対するプローブだけをやり直す**。他の5オポネント(TFT/AllC/AllD/
Random/WSLS)のプローブ結果・確証的結果(H1〜H3)は一切変更しない。

2段階のスクリプトを使う:

1. `scripts/reprobe_gradual_rule_fix.py` — 既存のcensusを読み、修正済みの
   Gradual・GradualAppeasementだけを再プローブして、**別ディレクトリ**に
   `seed_X_probe_age_G_noiseN_moves.csv.gz`/`_summary.csv`(この2オポネントの行のみ)を書く
   (元の本番データ`scripts/output/markov_sweep_production_noise0{1,0}/`は一切変更しない)。
2. `scripts/reanalyze_gradual_rule_fix.py` — 1の出力と、既に確定済みの記述的CSV
   (`markov_confirmatory_descriptive_{primary,sensitivity}.csv`)を読み、Gradual・
   GradualAppeasementの行だけを再計算して差し替えたCSVと、修正前後の比較CSVを書く
   (他5オポネントの行・確証的結果のJSONはそのままコピーされるだけで再計算しない)。

どちらも8節と同じ作業フォルダ(venv・raw probe dataが既にある場所)でそのまま使える。

### 9-1. 主条件(ノイズ0.01)の再プローブ・再分析

```bash
mkdir -p logs
nohup caffeinate -dimsu bash -c '
venv/bin/python scripts/reprobe_gradual_rule_fix.py \
  --data-dir scripts/output/markov_sweep_production_noise01 \
  --out-dir scripts/output/markov_sweep_production_noise01_gradual_reprobe \
  --probe-noise-rate 0.01 \
&& \
venv/bin/python scripts/reanalyze_gradual_rule_fix.py \
  --data-dir scripts/output/markov_sweep_production_noise01 \
  --reprobe-dir scripts/output/markov_sweep_production_noise01_gradual_reprobe \
  --probe-noise-rate 0.01 \
  --role primary \
  --original-descriptive-csv scripts/output/markov_confirmatory_production_noise01/markov_confirmatory_descriptive_primary.csv \
  --results-dir scripts/output/markov_confirmatory_production_noise01_gradual_fixed \
&& \
venv/bin/python scripts/reprobe_gradual_rule_fix.py \
  --data-dir scripts/output/markov_sweep_production_noise00 \
  --out-dir scripts/output/markov_sweep_production_noise00_gradual_reprobe \
  --probe-noise-rate 0.0 \
&& \
venv/bin/python scripts/reanalyze_gradual_rule_fix.py \
  --data-dir scripts/output/markov_sweep_production_noise00 \
  --reprobe-dir scripts/output/markov_sweep_production_noise00_gradual_reprobe \
  --probe-noise-rate 0.0 \
  --role sensitivity \
  --original-descriptive-csv scripts/output/markov_confirmatory_production_noise00/markov_confirmatory_descriptive_sensitivity.csv \
  --results-dir scripts/output/markov_confirmatory_production_noise00_gradual_fixed
' > logs/gradual_rule_fix_both.log 2>&1 &
disown
```

(主条件の再プローブ→再分析→感度分析の再プローブ→再分析、の順に1つのバックグラウンド
ジョブとして続けて実行する。`--data-dir`/`--moves-suffix`相当の`--probe-noise-rate`の
対応は8-3節と同じ: `markov_sweep_production_noise01/`は`0.01`、`noise00/`は`0.0`。
すべてのデフォルト値[`--seeds 70 --seed-offset 100 --window-start 1000 --window-end 5000
--probe-interval 50 --n-boot-descriptive 200 --n-boot-scheme-a 30 --workers 5`など]は
8節・事前登録と同じなので省略している。)

### 9-2. 所要時間・メモリの見積もり

パイロットデータ(クラウド)での実測([seed,age]単位の処理時間)と、8-4節で実測済みの
本番group_size=56でのブートストラップ1回あたりコスト(Gradual約1.09ms)を合成した見積もり:

- **再プローブ(`reprobe_gradual_rule_fix.py`)**: 純粋なゲームシミュレーション(ブートストラップなし)。
  パイロット実測で1 (seed,age)単位あたり約1.5秒。70 seed×81時点=5670単位 ÷ `--workers 5`
  ≈ **約28分/条件**。
- **再分析(`reanalyze_gradual_rule_fix.py`)**: 2オポネント分のスキームA(70 seed×30回)
  +スキームB(3群サイズ×200回)のブートストラップのみ(確証的検定のseedブートストラップは
  Gradual/GradualAppeasementには元々適用されないので無し)。計算コストはオポネント2つ分
  (8節の81時点・7オポネント分の約158秒/時点から比例按分)で約15秒/時点、raw moves読み込み
  I/O(オポネント2/7分なので元の87.5秒/時点から縮小)で約30秒/時点、合計約45秒/時点×81時点
  ÷ `--workers 5` ≈ **約12分/条件**。
- **合計**: 約40分/条件(再プローブ約28分+再分析約12分)、**両条件(主・感度分析)で
  約80分(約1.3時間)**(8節の本来の確証的分析の約1.5時間より短い — オポネント2つ分・
  確証的ブートストラップ無しのため)。
- メモリ:8-4節と同様、1GB未満。

本番前に実測で見積もりを更新したい場合は、8-4節と同じ要領で`--window-end`を縮めて
(例:`--window-end 1100`で3時点のみ)小規模に実行し、かかった時間を時点数で割る。

### 9-3. 進捗・再開

再プローブは`(seed, age)`単位でファイルの有無を見て再開する(8-5節と同様、同じコマンドを
もう一度打てばよい)。再分析は時点(age)単位で`<results-dir>/_age_cache_<role>_gradual_reprobe/`
にキャッシュする(`.gitignore`対象)。

```bash
ps aux | grep -E "reprobe_gradual|reanalyze_gradual" | grep -v grep
tail -40 logs/gradual_rule_fix_both.log
```

### 9-4. 結果の確認・図の再生成・push

再分析の結果(`scripts/output/markov_confirmatory_production_noise0{1,0}_gradual_fixed/`)には
以下が書かれる:
- `markov_confirmatory_descriptive_{primary,sensitivity}.csv` — 全オポネントの記述的CSV
  (Gradual・GradualAppeasementの行だけが修正後の値に差し替わっている)
- `markov_confirmatory_gradual_rule_fix_comparison_{primary,sensitivity}.csv` — 修正前後の
  値を並べた比較表(`docs/DEVIATION_gradual_rule.md`5項の「修正前の値も並べて報告する」)

8-6節と同じ3つの図スクリプトを、この新しい記述的CSVを指定して再実行すれば、Gradual・
GradualAppeasementの図だけが更新される(他オポネントの図も再生成されるが、入力データが
同じなので内容は変わらない):

```bash
venv/bin/python scripts/plot_markov_confirmatory_trends.py \
  --descriptive-csv scripts/output/markov_confirmatory_production_noise01_gradual_fixed/markov_confirmatory_descriptive_primary.csv \
  --role primary --out-dir scripts/output/figures
# plot_markov_confirmatory_sampling_comparison.py / analyze_markov_confirmatory_regime_correspondence.py も同様
# (--role sensitivity の場合はnoise01→noise00に読み替え)
```

pushするファイル(新規追加分。再プローブの生データ・再分析のageキャッシュは`.gitignore`対象):

```bash
git branch --show-current   # claude/markov-production-analysis を確認
git status --short
git add scripts/output/markov_confirmatory_production_noise01_gradual_fixed/markov_confirmatory_descriptive_primary.csv \
        scripts/output/markov_confirmatory_production_noise01_gradual_fixed/markov_confirmatory_gradual_rule_fix_comparison_primary.csv \
        scripts/output/markov_confirmatory_production_noise00_gradual_fixed/markov_confirmatory_descriptive_sensitivity.csv \
        scripts/output/markov_confirmatory_production_noise00_gradual_fixed/markov_confirmatory_gradual_rule_fix_comparison_sensitivity.csv \
        scripts/output/figures/markov_confirmatory_*.png
git commit -m "Gradual rule fix: re-probed Gradual/GradualAppeasement descriptive results (primary + sensitivity)"
git push -u origin claude/markov-production-analysis
```

---

## 10. 記憶の深さの網(探索的・事前登録外)

**対象ブランチ:`claude/markov-production-analysis`。** 人の論文(JOEP、N=171)と同じ方式・
同じ見た目で、3ステップのマルコフ網が「記憶の深さ」とともにどう変わるかを見る分析。
愛着の3群の代わりに、有効な記憶の深さ(`evopersonality_abm.lindgren.effective_memory_depth`、
表を振る舞いを変えずに縮められる最小の深さ)で3群(浅い・中・深い)に分ける。GradualはDEVIATION_gradual_rule.mdの
修正後の規則で再プローブしたデータ(`scripts/reprobe_gradual_rule_fix.py`の出力)を使う
— **9節の再プローブが済んでから**着手すること。

### 10-1. R用の入力ファイルの書き出し(Mac)

```bash
venv/bin/python scripts/export_depth_groups_for_r.py \
  --data-dir scripts/output/markov_sweep_production_noise01 \
  --gradual-reprobe-dir scripts/output/markov_sweep_production_noise01_gradual_reprobe \
  --moves-suffix noise0p01 \
  --out-dir scripts/output/markov_depth_groups
```

(すべてのデフォルト値[`--seeds 70 --seed-offset 100 --window-start 1000 --window-end 5000
--probe-interval 50 --group-size 56`など]は事前登録の主条件・分析窓に合わせてある。)

実行すると:
1. まず`scripts/output/markov_depth_groups/depth_distribution_by_generation.csv`に、
   世代ごとの表面上の深さ・有効な深さの分布(頻度重み付き)を書く。
2. 続けて、観測された有効な深さの値を3つの連続範囲(浅い・中・深い)に分ける区切りを、
   各群が`--group-size`(既定56)人分の**別々のseed**を確保できることを条件に自動提案し、
   標準出力に表示する(例:`proposed cutpoints: Low<=2, 3<=Mid<=3, High>=4`)。
   本番データでこの提案を見てから、必要なら`--depth-cutpoints "D1,D2"`で上書きして
   再実行できる(例:`--depth-cutpoints "2,4"`)。
3. 各群56人を選び(別々のseedを優先、足りない場合のみ同じseedの別時点を使う)、
   `scripts/output/markov_depth_groups/`に`TFT_depthLow.txt`・`GRA_depthMid.txt`などの
   5オポネント×3群=15個のテキストファイルと、各人のseed・世代・有効な深さの一覧
   (`depth_group_participants.csv`)を書く。

**所要時間の見積もり**(パイロットでの実測ベース):census走査(Phase 1)は70 seed×81時点で
約6分。系列抽出(Phase 2、並列化なし・単一プロセス)は選ばれた参加者が属する
(seed, 時点)の組ごとにraw movesファイルを1回ずつ読む必要があり、1ファイルあたり実測
約4.4秒 — 3群×56人=最大168組として**約12分**。**合計で約18分**(提案された区切りの
確認・`--depth-cutpoints`での再実行が必要な場合はその分追加)。メモリは1GB未満。

### 10-2. R の実行(Windows でも Mac でも)

1. `docs/r_scripts/GRA_MAR10feb04Fmarkov_network_with_loops_param_groups.R`・
   `docs/r_scripts/draw_all_opponents_depth_groups.R`・`docs/r_scripts/vocab/`の5つの
   予約語ファイルを、10-1の出力(`TFT_depthLow.txt`など15個のテキストファイル)と
   **同じフォルダ**にコピーする。
2. Rとパッケージ`igraph`・`tidyverse`がインストールされていることを確認する
   (`install.packages(c("igraph","tidyverse"))`)。
3. `draw_all_opponents_depth_groups.R`を開き、`work_dir`をそのフォルダの絶対パスに
   変更してから実行する(Rコンソールで`source("draw_all_opponents_depth_groups.R")`、
   またはコマンドラインで`Rscript draw_all_opponents_depth_groups.R`)。
4. TFT・GRA・RAN・ALC・ALDの5オポネント分、`<日時>_markov_network_3groups_<OPPONENT>_j030_with_loops.png`
   という名前のPNGが同じフォルダに出力される(各PNGは浅い・中・深い群の3パネル)。

(このRスクリプトは本セッションの環境にRが無いため実行確認していない — まず1オポネント分
[`GRA_MAR10feb04Fmarkov_network_with_loops_param_groups.R`単体を`OPPONENT <- "GRA"`のまま
手動で`source`]で問題なく動くことを確認してから、`draw_all_opponents_depth_groups.R`で
5オポネント分を一括実行することを推奨する。描画ロジック自体は元のOSFコードアーカイブの
スクリプトから一切変更していない[変更箇所はグループ名の3箇所のみ、コミットメッセージに
diffで検証済みと明記]。)

### 10-3. Python版の図(有効な深さ1〜6、Mac)

```bash
venv/bin/python scripts/plot_markov_depth_networks.py \
  --data-dir scripts/output/markov_sweep_production_noise01 \
  --gradual-reprobe-dir scripts/output/markov_sweep_production_noise01_gradual_reprobe \
  --moves-suffix noise0p01 \
  --out-dir scripts/output/figures
```

オポネントごとに`markov_depth_networks_<opponent>.png`(有効な深さ1〜6の6パネル、
Jaccard≥.30、自己ループあり、赤=6パネルすべてに共通の辺・青=その深さ特有の辺)を書く。

**所要時間の見積もり**:10-1と同じPhase 1(約6分)+ Phase 2(今回は深さ1〜6の6群、
最大6×56=336組の(seed,時点)ペア、単一プロセスで約4.4秒/ファイル ≈ **約25分**)。
**合計で約31分**。メモリは1GB未満。

### 10-4. 結果のpush

```bash
git branch --show-current   # claude/markov-production-analysis を確認
git status --short
git add scripts/output/markov_depth_groups/depth_distribution_by_generation.csv \
        scripts/output/markov_depth_groups/depth_group_participants.csv \
        scripts/output/figures/markov_depth_networks_*.png
git commit -m "Memory-depth network analysis: distribution report, participant lists, Python figures"
git push -u origin claude/markov-production-analysis
```

(`TFT_depthLow.txt`などのR入力テキストファイルと、Rが出力するPNGは、人の論文の実際の
3-gram系列相当のデータを含む中間生成物であり軽量なので任意でpushしてよいが、必須では
ない — 再現したい場合は10-1を再実行すれば同じ`--master-seed`から同じファイルが得られる。
Rで生成したPNGをpushする場合は`scripts/output/figures/`に移してから`git add`に加える。)

---

## 11. Jaccard閾値×世代のHTMLビューア(探索的・事前登録外)

**対象ブランチ:`claude/markov-production-analysis`。** 人(N=171)とABMの3ステップマルコフ網を、
Jaccard閾値と世代の2軸で動かしながら比較する、ブラウザで開ける1枚のHTMLビューア
(`scripts/output/figures/markov_edge_weight_viewer.html`)。**10節の記憶の深さの区切り
(`--depth-cutpoints`で提案・決定した値)が済んでから**着手すること。

### 11-1. 辺の重みの書き出し(Mac)

```bash
venv/bin/python scripts/export_markov_edge_weights_for_viewer.py \
  --data-dir scripts/output/markov_sweep_production_noise01 \
  --gradual-reprobe-dir scripts/output/markov_sweep_production_noise01_gradual_reprobe \
  --moves-suffix noise0p01 \
  --probe-noise-rate 0.01 \
  --depth-cutpoints "D1,D2" \
  --out-path scripts/output/figures/markov_edge_weights.json
```

(`--depth-cutpoints`は10節で使ったのと**同じ**値を指定する(浅い・中・深いの区切りを
両方の出力で揃えるため)。それ以外のデフォルト値[`--seeds 70 --seed-offset 100
--generations-start 0 --generations-end 5000 --probe-interval 50 --group-size 56
--n-draws 200 --workers 5`など]は依頼どおり。`--gradual-reprobe-dir`は9節の再プローブが
**gen 0〜5000の全時点**をカバーしていない場合[9節の既定はgen 1000〜5000のみ]、Gradualの
ネットワークはgen 1000未満の時点で欠落する[ビューア上は「データなし」として扱われる —
エラーにはならない]。gen 0〜5000全体でGradualも見たい場合は、9節の`reprobe_gradual_rule_fix.py`を
`--window-start 0 --window-end 5000`で追加実行してから、このスクリプトを実行すること。)

**所要時間の見積もり**(8-4節・10-1節の実測ベースの推定):101時点(gen 0〜5000、50世代おき)
×70 seed×2ファイル(主データ+Gradual再プローブ)を読む必要があり、1時点あたりのI/Oは
8-4節で確立した主データの約87.5秒に、Gradual再プローブ分(軽量、2オポネント分のみ)の
約30〜40秒を加えて約120〜130秒/時点。計算(5オポネント×4群×200回のブートストラップ)は
1時点あたり数秒と軽い。101時点÷`--workers 5` ≈ **約42〜45分**。年代ごとにpickleキャッシュ
するので(`_age_cache_edge_weights/`、`.gitignore`対象)、中断しても同じコマンドで再開できる。

**出力サイズの見積もり**(スクリプトのdocstringに詳細):64ノードのオポネントは1ノードあたり
最大4つの遷移先+自己ループ1つなので最大約5×64=320通りの辺。4つの群の作り方×5オポネント×
101時点×最大320通り ≈ 上限約65万行(多くの時点・群では閾値以前の段階でもっと少ない)。
JSONファイル全体でおよそ**10〜30MB**程度を見込む。

### 11-2. (任意)Rのノード配置をビューアに読み込ませる

デフォルトでは規則的な格子配置だが、論文と同じ力学的レイアウトを使いたい場合:

1. `docs/r_scripts/GRA_MAR10feb04Fmarkov_network_with_loops_export_layout.R`を、
   人の3群のテキストファイル・予約語ファイルと同じフォルダに置いて、オポネントごとに
   (`OPPONENT <- "TFT"`などに変更して)実行する — 各回`<OPPONENT>_layout.csv`
   (例:`TFT_layout.csv`、列: node,x,y)が出力される。
2. 5オポネント分の`<CODE>_layout.csv`(TFT・GRA・RAN・ALC・ALD)を
   `scripts/output/figures/`(`markov_edge_weight_viewer.html`と同じフォルダ)にコピーする。
   ビューアは起動時に自動でこれらを探し、見つかった分だけ使う(無ければ格子配置にフォール
   バックする、エラーにはならない)。

### 11-3. HTMLの開き方

ビューアは起動時に同じフォルダの`markov_edge_weights.json`を`fetch`で読み込むため、
**HTMLファイルを`file://`で直接ダブルクリックして開くと、ブラウザのCORS制限により
読み込みに失敗する**(ブラウザのJavaScriptコンソールにエラーが出る)。
`scripts/output/figures/`フォルダで簡易ローカルサーバーを立てて開くこと(外部には一切
通信しない、同じマシン内のローカルサーバー):

```bash
cd scripts/output/figures
python3 -m http.server 8000
```

ブラウザで`http://localhost:8000/markov_edge_weight_viewer.html`を開く(Mac・Windows
どちらでも、ブラウザとPython 3があれば同じ手順でよい)。使い終わったら`Ctrl+C`でサーバーを
止める。

### 11-4. 結果のpush

```bash
git branch --show-current   # claude/markov-production-analysis を確認
git status --short
git add scripts/output/figures/markov_edge_weights.json \
        scripts/output/figures/markov_edge_weight_viewer.html \
        scripts/output/figures/*_layout.csv
git commit -m "Markov edge-weight viewer: production edge-weight JSON (+ optional R layouts)"
git push -u origin claude/markov-production-analysis
```

(`markov_edge_weights.json`は10〜30MB程度見込みなので、pushに時間がかかる場合がある。
`_age_cache_edge_weights/`は`.gitignore`対象なので`git status --short`に出ないはず。)

### 11-5. (任意)ノードの出現率の追加書き出し(探索的・事前登録外)

`--nodes-only`を付けて**同じ引数で**再実行すると、辺の重みではなく、各3-gramの出現率
(その3-gramを含む仮想参加者の割合、200回の抽出で平均)を書き出す。**11-1の辺の重みの
書き出しと完全に同じ乱数の流れ(同じ`--master-seed`・世代ごとの乱数・同じ抽出順序)を
使うので、11-1と同じ引数で実行すれば、既にpush済みの`markov_edge_weights.json`が使った
のと同一の200回の抽出から出現率を計算する**(`markov_edge_weights.json`自体は一切
読み書きしない — 値は変わらない)。

```bash
venv/bin/python scripts/export_markov_edge_weights_for_viewer.py \
  --data-dir scripts/output/markov_sweep_production_noise01 \
  --gradual-reprobe-dir scripts/output/markov_sweep_production_noise01_gradual_reprobe \
  --moves-suffix noise0p01 \
  --probe-noise-rate 0.01 \
  --depth-cutpoints "D1,D2" \
  --nodes-only \
  --out-path scripts/output/figures/markov_edge_weights.json
```

(`--depth-cutpoints`は11-1と同じ値を指定する。`--out-path`も11-1と同じ値を渡すが、
`--nodes-only`時はこのパスは読み書きされず、代わりにその**同じフォルダ**に
`markov_node_freq.json`が書かれる。キャッシュは`_age_cache_node_freq/`で、
11-1の`_age_cache_edge_weights/`とは別。)

所要時間は11-1とほぼ同じ(同じI/O・同じ回数のブートストラップ抽出を行うため)。

pushするファイル:

```bash
git add scripts/output/figures/markov_node_freq.json
git commit -m "Add ABM node-appearance-rate JSON for the Markov network viewer"
git push -u origin claude/markov-production-analysis
```

---

## 12. SELF/POP/WSLS自己対戦プローブ・底からの回復時間(探索的・事前登録外)

**対象ブランチ:`claude/markov-production-analysis`。**
`docs/EXPLORATORY_selfplay_recovery.md`の調査(進化した個体がTFT相手には底底底から戻れないのが
「回復力の欠如」か「仲間限定の回復力」かを見分ける)の本番実行。既存の主条件データ
(`scripts/output/markov_sweep_production_noise01/`、8-9節で確定済み)のcensus(SELF/POP用)と
既存のWSLSプローブ記録(そのまま再利用)だけを使い、**新たな進化は行わない**。
`scripts/output/figures/markov_edge_weights.json`・`markov_node_freq.json`は一切読み書きしない
(別ファイルに書く)。

```bash
mkdir -p logs
nohup caffeinate -dimsu venv/bin/python scripts/export_markov_selfplay_recovery.py \
  --data-dir scripts/output/markov_sweep_production_noise01 \
  --moves-suffix noise0p01 \
  --noise-rate 0.01 \
  > logs/selfplay_recovery.log 2>&1 &
disown
```

(`--seeds 70 --seed-offset 100 --generations-start 1000 --generations-end 5000
--probe-interval 50 --probe-repeats 30 --probe-rounds 100 --main-analysis-rounds 20
--group-size 56 --n-draws 200 --workers 5`は既定値のまま。50世代おきが重すぎる場合は
`--probe-interval 250`に切り替える[スクリプト自身のヘルプにも明記]。)

### 12-1. 所要時間・メモリの見積もり

クラウドのパイロットデータ(`scripts/output/markov_sweep_pilot_noise00`、10 seed)に対する
実測ベースの推定:本番と同じ`--probe-repeats 30 --probe-rounds 100 --main-analysis-rounds 20`
で1 (seed, age)単位あたり約2.9秒(SELF/POP series構築+底からの回復時間の計算を合わせた値;
回復時間の追加分は、多くの対戦が20回以内の早い手番で回復してしまうため計算量への影響は小さい)。
70 seed×81時点(gen 1000〜5000、50世代おき)÷`--workers 5` ≈ **約55分**。
(ブートストラップ集計[widest path・辺の重み・ノード出現率]は1時点あたり数秒と軽く、
上記の見積もりに対して無視できる。)年代ごとにpickleキャッシュするので
(`_age_cache_selfplay/`、`.gitignore`対象)、中断しても同じコマンドで再開できる。
メモリは9-2節と同様、1GB未満の見込み。

本番前に実測で見積もりを更新したい場合は、8-4節と同じ要領で`--generations-end`を縮めて
(例:`--generations-end 1100`で3時点のみ)小規模に実行し、かかった時間を時点数で割る。

### 12-2. 進捗・再開

```bash
ps aux | grep export_markov_selfplay_recovery | grep -v grep
tail -40 logs/selfplay_recovery.log
```

### 12-3. 結果の確認・push

書き出されるファイル:
- `scripts/output/figures/markov_selfplay_widest_path.csv`
- `scripts/output/figures/markov_selfplay_recovery_time.csv`
- `scripts/output/figures/markov_selfplay_edges.json`
- `scripts/output/figures/markov_selfplay_nodes.json`

```bash
git branch --show-current   # claude/markov-production-analysis を確認
git status --short
git add scripts/output/figures/markov_selfplay_widest_path.csv \
        scripts/output/figures/markov_selfplay_recovery_time.csv \
        scripts/output/figures/markov_selfplay_edges.json \
        scripts/output/figures/markov_selfplay_nodes.json
git commit -m "SELF/POP/WSLS self-play recovery probe: widest-path, recovery-time, edge/node JSON"
git push -u origin claude/markov-production-analysis
```

(**`scripts/output/figures/markov_edge_weights.json`・`markov_node_freq.json`はこの手順で
一切変更されないはず**——`git status --short`でこの2ファイルが変更として出てこないことを
確認すること。`_age_cache_selfplay/`は`.gitignore`対象なので出ないはず。)
