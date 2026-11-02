# Kモデル Markov構造・本番ラン(70 seed×5000世代)をMacで実行する

事前登録済み(`docs/PREREGISTRATION_k_model_markov.md`、確定コミット:
そのファイルの冒頭を参照)。パイロット(seed 100〜103、1200世代)は
クラウド環境で完走・分析済み(`docs/K_MODEL_MARKOV_DESIGN.md` 8節)。
本番は規模が大きく、クラウド環境の2時間バックグラウンド上限を何十回も
再開するより、ユーザー自身のMacで`nohup`実行する方が実際的。このファイルは
そのための手順書(`docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md`と
同じ形式)。

対象ブランチ: `claude/k-model-markov-probe`
対象リポジトリ: `dpforest610/EvoPersonality_ABM`
対象スクリプト: `scripts/run_coupling_markov_pilot.py`(進化+プローブ、本番規模の
パラメータにも既に対応済み——名前は「pilot」だが汎用)、
`scripts/compute_k_model_type_thresholds.py`(しきい値)、
`scripts/run_k_model_confirmatory_analysis.py`(確証的分析、置換検定を含む)、
`scripts/check_coupling_markov_progress.py`(進捗確認)、
`scripts/analyze_k_model_permutation_null_center.py`(9節、置換検定の帰無分布の
中心——事前登録外の探索的分析)。

---

## 0. 本番の設定(事前登録3節、固定)

| パラメータ | 値 |
|---|---|
| seed | 200〜269(70本) |
| population_size | 100 |
| generations | 5000 |
| rounds_per_duel | 50 |
| point_mutation_rate | 0.05 |
| duplication_rate | 0.02 |
| split_rate | 0.02 |
| initial_memory_depth | 1 |
| max_memory_depth | 8 |
| min_memory_depth | 0 |
| k_mutation_std | 0.05 |
| evolution_noise_std(進化本体) | 0.05 |
| probe_noise_std(プローブ中のKの層) | 0.05 |
| probe_start / probe_interval | 1000 / 50(gen 1000〜5000、81時点) |
| probe_repeats | 30 |
| probe_rounds | 100 |
| free_rounds(回復時間の自由対戦) | 20 |

出力先: `scripts/output/coupling_markov_production`(新規——既存の
`scripts/output/coupling_markov_pilot`は書き換えない)。

---

## 1. 準備

### 1-1. Python・依存パッケージ・clone

`docs/LOCAL_RUN.md`の1-1〜1-3と同じ:

```bash
brew install python@3.11 git   # Homebrewが無ければ https://brew.sh を先に
git clone https://github.com/dpforest610/EvoPersonality_ABM.git
cd EvoPersonality_ABM
git checkout claude/k-model-markov-probe
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

(以後、新しいターミナルを開くたびに`cd EvoPersonality_ABM`だけでよい——以下の
コマンドはすべて`venv/bin/python`を直接指定するので、`source venv/bin/activate`
は必須ではないが、していても構わない。)

### 1-2. テスト

```bash
venv/bin/python -m pytest -q
```

全件passedになることを確認(1件でもfailしたら、2節に進まずまず報告して)。

---

## 2. 小規模の実測(世代を1100程度に絞る)

パイロットの実測から外挿した見積もり(事前登録9節:1 seedあたり約2.1時間、
70 seedを`--workers 5`で約29時間)は、クラウド環境のベンチマークに基づく
外挿であり、実測ではない。本番を起動する前に、Mac上で実際に計測し、
この見積もりを更新する。

```bash
mkdir -p /tmp/coupling_markov_timing_check
time venv/bin/python scripts/run_coupling_markov_pilot.py \
  --seeds 1 --seed-offset 999 --population-size 100 --generations 1100 \
  --rounds-per-duel 50 --initial-memory-depth 1 --point-mutation-rate 0.05 \
  --duplication-rate 0.02 --split-rate 0.02 --max-memory-depth 8 --min-memory-depth 0 \
  --k-mutation-std 0.05 --evolution-noise-std 0.05 --probe-noise-std 0.05 \
  --probe-start 1000 --probe-interval 50 --probe-repeats 30 --probe-rounds 100 \
  --free-rounds 20 --workers 1 \
  --out-dir /tmp/coupling_markov_timing_check
```

(seed 999は本番の200〜269と重複しない捨てseed。`--workers 1`は1 seedしか
回さないので関係ない。)

表示された実時間(`real`)を使って、1 seedあたりの時間を更新する:
世代1100のうち、プローブはgen 1000・1050・1100の3時点(probe_repeats=30・
probe_rounds=100・8相手)が走る。実測した`real`秒数をそのまま
「1 seedあたり約X秒(世代1100相当)」として記録し、5000世代・81時点に
単純に伸ばした値(進化部分は世代数に比例、プローブ部分は時点数におおむね
比例)を2節の見積もりの代わりに使う。70 seedを`--workers 5`(6コア−1)で
割った実行時間を見積もり直してから、3節に進む。

終わったら削除してよい: `rm -rf /tmp/coupling_markov_timing_check`。

---

## 3. 本番の起動

### 3-1. コア数の確認

```bash
sysctl -n hw.ncpu
```

6コアなら`--workers 5`(1コアはOS・他作業用に残す)。コア数が違う場合は
`workers = コア数 - 1`に読み替える。

### 3-2. バックグラウンド起動

```bash
mkdir -p logs
nohup caffeinate -dimsu venv/bin/python scripts/run_coupling_markov_pilot.py \
  --seeds 70 --seed-offset 200 --population-size 100 --generations 5000 \
  --rounds-per-duel 50 --initial-memory-depth 1 --point-mutation-rate 0.05 \
  --duplication-rate 0.02 --split-rate 0.02 --max-memory-depth 8 --min-memory-depth 0 \
  --k-mutation-std 0.05 --evolution-noise-std 0.05 --probe-noise-std 0.05 \
  --probe-start 1000 --probe-interval 50 --probe-repeats 30 --probe-rounds 100 \
  --free-rounds 20 --workers 5 --push \
  --out-dir scripts/output/coupling_markov_production \
  > logs/coupling_markov_production.log 2>&1 &
disown
```

(`--push`:1 seedが完全に完走した時だけ、その軽量な`generation_stats.csv`を
自動でコミット・push する——6節で説明するとおり、生の記録[census/moves/
recovery]はpushされない。`caffeinate -dimsu`でディスプレイ・システムスリープを
防止——電源アダプタ接続時は蓋を閉じてもスリープしない設定にしておくと
より安全[システム設定→ロック画面、または省エネルギー]。`disown`で、
ターミナルを閉じてもプロセスは生き続ける。)

---

## 4. 進み具合の確認

```bash
venv/bin/python scripts/check_coupling_markov_progress.py \
  --out-dir scripts/output/coupling_markov_production --n-seeds 70 --seed-offset 200 --generations 5000
```

出力例:

```
scripts/output/coupling_markov_production  done: 12/70  in-progress (seed:gen/total): 212:3400/5000, 213:150/5000
```

プロセスが生きているかの確認・ログの末尾:

```bash
ps aux | grep run_coupling_markov_pilot | grep -v grep
tail -40 logs/coupling_markov_production.log
```

---

## 5. 止まったときの再開

### 5-1. 安全に停止する

```bash
pkill -f run_coupling_markov_pilot.py
```

**チェックポイントは50世代おき(プローブと同じタイミング)に保存されるので、
killされた時点までに失われるのは、直前のチェックポイント以降の最大50世代分
だけ**。`ps aux`で消えたことを確認する。

### 5-2. Mac再起動・スリープ・停電の後、再開する

**3-2のコマンドを、全く同じ内容でもう一度打つだけ。** 完了済みのseedは
`seed_X_generation_stats.csv`の行数(=世代数)が`--generations`と一致していれば
スキップされ、未完了のseedは`seed_X_checkpoint.pkl`から再開する
(`scripts/run_coupling_markov_pilot.py`自身のテストで、中断・再開した結果が
中断せず完走した場合とビット一致することを確認済み)。念のため4節のコマンドで
確認する。

---

## 6. 結果の回収とpush

**リポジトリに入れるもの(軽量・集計済み)**:
- `seed_X_generation_stats.csv`(3-2の`--push`で自動コミット・push済み)
- このあとの手順で作る`scripts/output/coupling_markov_production/analysis/`
  (しきい値)と`scripts/output/k_model_confirmatory_production/`
  (確証的分析の結果、8節)

**Macにだけ残すもの(生の記録、`.gitignore`対象——pushしない)**:
- `seed_X_probe_age_G_census.csv.gz`・`..._moves.csv.gz`・`..._recovery.csv.gz`
  (`.gitignore`の`scripts/output/coupling_markov_*/*_{census,moves,recovery}.csv.gz`
  パターンに一致)
- `seed_X_checkpoint.pkl`(未完了seedの再開用、完走後は自動削除)
- `scripts/output/k_model_confirmatory_production/_age_cache/`(確証的分析の
  per-age再開用キャッシュ、8節)

70/70完了を4節のコマンドで確認したら、念のため押し漏れがないか確認する:

```bash
git status --short
git log --oneline -10
```

何か残っていれば(生の記録ファイルは`.gitignore`対象なので対象外):

```bash
git add scripts/output/coupling_markov_production/
git commit -m "K-model Markov production: manual checkpoint of completed units"
git push
```

---

## 7. トラブル時

| 症状 | 対処 |
|---|---|
| `ModuleNotFoundError: No module named 'numpy'`(など) | `venv/bin/python`を使っているか確認(`source venv/bin/activate`を忘れていても`venv/bin/python`を直接呼べば動く) |
| `python3 scripts/...`で`No such file or directory` | リポジトリのルート(`EvoPersonality_ABM/`)にいない。`cd`し直す |
| ディスク容量不足(`No space left on device`) | `df -h .`で空き容量確認。6節の「Macにだけ残すもの」が本番規模(70 seed×81時点)で数十GBになる見込み。パイロット比(4 seed×5時点)からの外挿で見積もり直してから空き容量を確保する |
| `git push`が`rejected` | `git pull --rebase`してから再度`git push`。コンフリクトが出た場合、`scripts/output/`以下のCSV/JSONはこちらが勝って問題ないことが多いが、判断に迷ったら触らずに相談する |
| Macがスリープして止まっていた | `caffeinate`を付け忘れていないか確認。5-2の手順で再開すれば、チェックポイントからの再開で済む |
| 蓋を閉じたら止まっていた | システム設定で「電源アダプタ接続時は蓋を閉じてもスリープしない」を有効にする。再開は5-2と同じ |

---

## 8. 確証的な分析の実行(事前登録4〜5節、置換検定を含む)

70/70完了を確認した後に実行する。**同じ作業フォルダでそのまま続けてよい**
(生の記録はローカルに既にあるので、別cloneは不要)。

### 8-1. しきい値の計算(事前登録3節)

```bash
venv/bin/python scripts/compute_k_model_type_thresholds.py \
  --data-dir scripts/output/coupling_markov_production --seeds 70 --seed-offset 200 \
  --min-generation 1000 \
  --out-dir scripts/output/coupling_markov_production/analysis
```

`type_thresholds.json`(本番データで一度だけ計算・固定するしきい値)と
`type_thresholds_by_generation.csv`(世代ごとの感度分析用)が書かれる。
これはリポジトリに入れる(6節):

```bash
git add scripts/output/coupling_markov_production/analysis/
git commit -m "K-model production: type thresholds"
git push
```

### 8-2. 小規模の実測(確証的分析の所要時間)

事前登録5-4-1節の見積もり(H-Aの点推定≈15分、置換検定[置換1000回×
draw50]≈12.5時間、seedブートストラップ[2000回]≈数十分、H-B/H-Cはどれも
軽い——合計でおおむね13〜14時間程度と見込まれるが、これもクラウド環境の
ベンチマークからの外挿であり実測ではない)を、本番データの一部(3時点だけ)
で実測してから、本番(81時点)の見積もりを更新する:

```bash
time venv/bin/python scripts/run_k_model_confirmatory_analysis.py \
  --data-dir scripts/output/coupling_markov_production --seeds 70 --seed-offset 200 \
  --thresholds-json scripts/output/coupling_markov_production/analysis/type_thresholds.json \
  --window-start 1000 --window-end 1100 \
  --n-draws-main 1000 --n-bootstrap-seed 2000 --n-permutations 1000 --permutation-draws-ha 50 \
  --workers 5 --results-dir /tmp/k_model_confirmatory_timing_check
```

(`--window-end 1100`でgen 1000・1050・1100の3時点だけ処理する。表示された
`real`秒数を3で割り、81時点分に単純に伸ばした値を本番の見積もりとする。)

終わったら削除してよい: `rm -rf /tmp/k_model_confirmatory_timing_check`。

### 8-3. 本番実行

```bash
mkdir -p logs
nohup caffeinate -dimsu venv/bin/python scripts/run_k_model_confirmatory_analysis.py \
  --data-dir scripts/output/coupling_markov_production --seeds 70 --seed-offset 200 \
  --thresholds-json scripts/output/coupling_markov_production/analysis/type_thresholds.json \
  --window-start 1000 --window-end 5000 \
  --n-draws-main 1000 --n-bootstrap-seed 2000 --n-permutations 1000 --permutation-draws-ha 50 \
  --workers 5 --results-dir scripts/output/k_model_confirmatory_production \
  > logs/k_model_confirmatory_production.log 2>&1 &
disown
```

(事前登録5-4節で確定した置換検定の設計——H-Aは置換1000回×1置換あたり
draw50回[本物のラベルの統計量も同じdraw50回で計算し直す]、H-B/H-Cは
置換1000回・drawという制約なし——がそのままデフォルト値になっている。)

### 8-4. 進捗・再開

時点(age)ごとに`<results-dir>/_age_cache/age_<N>.pkl`へ結果をキャッシュする
(`.gitignore`対象)。`pkill`や停電で止まっても、**同じコマンドをもう一度打てば**、
キャッシュ済みの時点はスキップして再開する。

```bash
ps aux | grep run_k_model_confirmatory_analysis | grep -v grep
tail -40 logs/k_model_confirmatory_production.log
ls scripts/output/k_model_confirmatory_production/_age_cache/ | wc -l   # 完了済み時点数(81で全完了)
```

### 8-5. 結果の確認・push

完了すると`scripts/output/k_model_confirmatory_production/k_model_confirmatory_summary.json`
が書かれ、H-A・H-B・H-Cそれぞれの比較・片側p値・Holm補正後の判定・置換検定の
p値が入る(ログの最後にも各仮説の判定が表示される)。これはリポジトリに入れる
(`_age_cache/`は`.gitignore`対象なので含まれない):

```bash
git add scripts/output/k_model_confirmatory_production/k_model_confirmatory_summary.json
git commit -m "K-model production: confirmatory analysis results (H-A/H-B/H-C, permutation test)"
git push
```

**この結果の中身(どの仮説が支持されたか)は、まだ解釈しない。** 事前登録の
手順どおりに出た生の判定を、次の段階で安念が確認する。

---

## 9. 置換検定の帰無分布の中心(事前登録外、探索的)

`docs/K_MODEL_MARKOV_RESULTS.md`3節のための分析。置換検定(8節)はラベルを
**各seedの中で**入れ替えるため、帰無分布は必ずしも0を中心にしない(系統間の
違いそのものが帰無分布にも残る)。置換検定のp値は本物の値が帰無分布の
どこに位置するかしか示さないので、帰無分布の中心(平均)そのものと、
本物の値との差(「型そのものの分」)を別途計算する。

8節の`_age_cache/`(時点ごとにキャッシュされた、置換ごとの生の値)を読む
ので、**8節の確証的分析を実行したのと同じMac・同じ`--results-dir`で**実行する
(`_age_cache/`は`.gitignore`対象でpushされていないため、Mac以外では読めない)。

```bash
venv/bin/python scripts/analyze_k_model_permutation_null_center.py \
  --results-dir scripts/output/k_model_confirmatory_production \
  --window-start 1000 --window-end 5000 \
  --out-csv scripts/output/k_model_confirmatory_production/permutation_null_center.csv
```

H-A・H-B・H-Cのすべての比較(11行)について、`true_diff_window_avg`(本物の差)・
`null_center`(置換の帰無分布の中心)・`effect_type_only`(その差)・
`n_permutations_used`を1行ずつ出力する。小さいスクリプトなので数秒〜数十秒で
終わる(81時点分の`_age_cache`を読むだけで、新しい抽出・置換は行わない)。

結果はリポジトリに入れる:

```bash
git add scripts/output/k_model_confirmatory_production/permutation_null_center.csv
git commit -m "K-model production: permutation null-center analysis (exploratory, section 3)"
git push
```

書かれたCSVの値を`docs/K_MODEL_MARKOV_RESULTS.md`3節の表に転記する
(この文書を書いた時点では`_age_cache`にアクセスできず、表は未記入のまま
残している)。

---

## 10. 学会・口頭発表用の図(ヒラギノ角ゴ)

8節の確証的分析(summary.json)とは別に、提示用の図を2枚作る。生のcensus/recovery
データ(census.csv.gz・recovery.csv.gz、Macにしかない)を読むので、8節と同じMacで
実行する(moves.csv.gzは読まないので、8節より高速)。

```bash
venv/bin/python scripts/plot_k_model_recovery_figures.py \
  --data-dir scripts/output/coupling_markov_production --seeds 70 --seed-offset 200 \
  --thresholds-json scripts/output/coupling_markov_production/analysis/type_thresholds.json \
  --window-start 1000 --window-end 5000 --n-bootstrap-seed 2000 \
  --out-dir scripts/output/figures
```

Macに実際にHiraginoフォント(Hiragino Sans / Hiragino Kaku Gothic ProN /
Hiragino Kaku Gothic Pro のいずれか)が入っていることを確認してから実行する
(入っていなければIPAGothicにフォールバックし、警告を標準エラーに出す)。

書かれる2枚:
- `k_model_recovery_rate_by_generation_SELF.png`:H-B、SELF相手の
  「20回のうちに和和和へ戻れた割合」を世代ごとに、K-安定型・K-拒絶型・
  K-とらわれ型・K-恐れ型の4類型で並べた折れ線図(seedをまたいだブートストラップ
  2000回の95%帯付き)。
- `k_model_recovery_rate_by_depth_stratum_SELF.png`:H-C、記憶の深さの3層
  (depthLow/depthMid/depthHigh)ごとに、K-安定型とK-恐れ型の戻れた割合
  (gen 1000〜5000平均)を並べた棒グラフ(同じブートストラップの95%誤差線)。

どちらの図にも「中央値で無理やり分けた型であり、集団は連続した一つの雲」の
注記が入っている。

リポジトリに入れる:

```bash
git add scripts/output/figures/k_model_recovery_rate_by_generation_SELF.png
git add scripts/output/figures/k_model_recovery_rate_by_depth_stratum_SELF.png
git commit -m "K-model production: conference figures (H-B by generation, H-C by depth stratum)"
git push
```

---

## 11. ビューア用データの書き出し(事前登録外、探索的——Lindgren側のビューアと同じ形式)

`scripts/export_k_model_markov_viewer.py`を本番用に直した(`scripts/
run_k_model_confirmatory_analysis.py`のH-Aと同じ系統間抽出+
`docs/DEVIATION_k_model_group_size.md`の人数規則、全8相手、K-安定型・
K-拒絶型・K-とらわれ型・K-恐れ型・合成Mixの5群)。8節の確証的分析とは
別物(置換検定もブートストラップも行わない、純粋に辺の重み・ノードの
出現率を集計するだけ)なので、8節の完了を待たずに並行して実行してよい。

### 11-1. 小規模の実測

```bash
time venv/bin/python scripts/export_k_model_markov_viewer.py \
  --data-dir scripts/output/coupling_markov_production --seeds 70 --seed-offset 200 \
  --thresholds-json scripts/output/coupling_markov_production/analysis/type_thresholds.json \
  --window-start 1000 --window-end 1100 --n-draws 200 --workers 5 \
  --out-dir /tmp/k_model_viewer_timing_check
```

(gen 1000・1050・1100の3時点だけ処理する。表示された`real`秒数を3で割り、
81時点分に単純に伸ばした値を本番の見積もりとする——クラウド環境での
ごく小規模なベンチマークからの外挿では、1時点あたり数十秒程度
[`--workers 5`で並列化すれば81時点全体で10〜20分程度]と見込まれるが、
実測ではないので必ずこの手順で更新すること。)

終わったら削除してよい: `rm -rf /tmp/k_model_viewer_timing_check`。

### 11-2. 本番実行

```bash
mkdir -p logs
nohup caffeinate -dimsu venv/bin/python scripts/export_k_model_markov_viewer.py \
  --data-dir scripts/output/coupling_markov_production --seeds 70 --seed-offset 200 \
  --thresholds-json scripts/output/coupling_markov_production/analysis/type_thresholds.json \
  --window-start 1000 --window-end 5000 --n-draws 200 --workers 5 \
  --out-dir scripts/output/coupling_markov_production/viewer \
  > logs/k_model_viewer_export.log 2>&1 &
disown
```

### 11-3. 進捗・再開

時点(age)ごとに`<out-dir>/_age_cache/age_<N>.pkl`へ結果をキャッシュする
(`.gitignore`の`scripts/output/**/_age_cache/`パターンで、この
`viewer/_age_cache/`も既に対象)。`pkill`や停電で止まっても、**同じコマンドを
もう一度打てば**キャッシュ済みの時点をスキップして再開する。

```bash
ps aux | grep export_k_model_markov_viewer | grep -v grep
tail -40 logs/k_model_viewer_export.log
ls scripts/output/coupling_markov_production/viewer/_age_cache/ | wc -l   # 完了済み時点数(81で全完了)
```

### 11-4. 結果の確認・push

完了すると次が書かれる:`k_model_type_counts.csv`・`k_model_widest_path.csv`・
`k_model_recovery_time.csv`・`k_model_effective_group_size.csv`(型ごと・
時点ごとに実際使った人数——合成Mix・K-拒絶型・K-とらわれ型も含め、
`docs/DEVIATION_k_model_group_size.md`と同じ規則)・`k_model_edges.json`・
`k_model_nodes.json`。いずれも集計済みの軽量ファイルなのでリポジトリに
入れる(`_age_cache/`は除く):

```bash
git add scripts/output/coupling_markov_production/viewer/k_model_type_counts.csv
git add scripts/output/coupling_markov_production/viewer/k_model_widest_path.csv
git add scripts/output/coupling_markov_production/viewer/k_model_recovery_time.csv
git add scripts/output/coupling_markov_production/viewer/k_model_effective_group_size.csv
git add scripts/output/coupling_markov_production/viewer/k_model_edges.json
git add scripts/output/coupling_markov_production/viewer/k_model_nodes.json
git commit -m "K-model production: viewer export (edges/nodes, all 8 opponents x 5 groupings x 81 ages)"
git push
```

---

## 12. ビューアを開く(`k_model_network_viewer.html`)

`scripts/output/figures/k_model_network_viewer.html`は、`k_model_edges.json`・
`k_model_nodes.json`と同じフォルダから`fetch()`で読むので、`file://`で直接
開くとブラウザのセキュリティ制限で読み込めない。簡易サーバー経由で開く。

```bash
cp scripts/output/figures/k_model_network_viewer.html \
   scripts/output/coupling_markov_production/viewer/index.html
cd scripts/output/coupling_markov_production/viewer
python3 -m http.server 8000
```

ブラウザで `http://localhost:8000/index.html` を開く。見終わったら
`Ctrl+C`でサーバーを止めてよい(`index.html`は`.gitignore`済みの`viewer/`内
コピーなのでpushされない——オリジナルは`scripts/output/figures/
k_model_network_viewer.html`のまま)。

操作:
- **相手**:TFT・GRA(Gradual)・RAN(Random)・ALC(AllC)・ALD(AllD)・WSLS・
  SELF(仲間)・POP(同じ集団)の8つ。後の3つは人のデータが無いため、人の
  3群のパネルは空になる(想定どおり)。
- **Kモデルの群**:K-安定型・K-拒絶型・K-とらわれ型・K-恐れ型・合成Mixの
  5つ(右上のパネルだけが切り替わる。人の3群のパネルは常にSecure/Mix/
  Fearful)。
- **Jaccard閾値**・**世代**の2つのスライダー(「再生」ボタンで自動再生)。
- ノードの大きさ=出現率(Kモデルのパネルのみ——人のパネルは出現率の
  データが無いため固定サイズ)。ノードの位置は64語彙全体の共通格子で
  固定(相手を切り替えても動かない)。
- 固着回路(Table 7)は、現在の閾値ですべての辺が揃っているときだけ
  色が付く。WSLS/SELF/POPには固着回路の定義が無いため、常に無色。

図には「中央値で無理やり分けた型であり、集団は連続した一つの雲」の
注記が常に表示されている。**探索的・事前登録外の可視化であり、
確証的結果(`docs/K_MODEL_MARKOV_RESULTS.md`)の判定根拠ではない。**

---

## 13. 群の人数の影響を確かめる(事前登録外、探索的)

K-安定型(目標56)・合成Mix(目標46)・K-恐れ型(目標69)は、系統間抽出の
目標人数自体が違う。その違いが特定の値にどれだけ効くかを見るため、
`scripts/export_k_model_equal_group_size_circuits.py`(新規)は、この3群を
**すべて同じ人数(46人。その時点でこの3群に共通して確保できる最小値
まで縮める)**で抽出した場合の値と、**今までどおり人の群に合わせた人数**
(56/46/69、各群ごとに縮める)で抽出した場合の値を、同じ時点・同じ列に
並べて出す。8節の確証的分析・11節のビューア用データとは別の、独立した
探索的スクリプト(どちらかが未実行でも単独で動く)。

計算する値(TFT・Gradual・AllC・AllDの4相手、81時点、3群それぞれ):
- TFT・Gradualの底底底→和和和のwidest path
- AllCの搾搾搾→和和和のwidest path
- AllDの底底底→惨惨惨のwidest path
- TFT・Gradual・AllCの協調の回路(和和和→和和搾→…)のj*
- Gradualの搾惨搾⇔惨搾惨の回路(TFT自身のTable 7の回路定義を、
  Gradualの網に当てはめて計算したもの——Gradual自身のTable 7の回路
  一覧にはこの回路は入っていない)のj*

### 13-1. 小規模の実測

```bash
time venv/bin/python scripts/export_k_model_equal_group_size_circuits.py \
  --data-dir scripts/output/coupling_markov_production --seeds 70 --seed-offset 200 \
  --thresholds-json scripts/output/coupling_markov_production/analysis/type_thresholds.json \
  --window-start 1000 --window-end 1100 --n-draws 200 --workers 5 \
  --out-csv /tmp/k_model_equal_group_size_circuits_timing_check.csv
```

(gen 1000・1050・1100の3時点だけ処理する。表示された`real`秒数を3で割り、
81時点分に単純に伸ばした値を本番の見積もりとする。11節の書き出し[全8相手×
5群]よりずっと軽い[4相手×3群のみ]ので、数分程度で終わると見込まれるが、
実測ではないので必ずこの手順で更新すること。)

終わったら削除してよい: `rm -f /tmp/k_model_equal_group_size_circuits_timing_check.csv`。

### 13-2. 本番実行

```bash
mkdir -p logs
nohup caffeinate -dimsu venv/bin/python scripts/export_k_model_equal_group_size_circuits.py \
  --data-dir scripts/output/coupling_markov_production --seeds 70 --seed-offset 200 \
  --thresholds-json scripts/output/coupling_markov_production/analysis/type_thresholds.json \
  --window-start 1000 --window-end 5000 --n-draws 200 --workers 5 \
  --out-csv scripts/output/coupling_markov_production/viewer/k_model_equal_group_size_circuits.csv \
  > logs/k_model_equal_group_size_circuits.log 2>&1 &
disown
```

### 13-3. 進捗・再開

時点ごとに`<出力CSVのあるフォルダ>/_age_cache_equal_group_size/age_<N>.pkl`へ
結果をキャッシュする(`.gitignore`の`scripts/output/**/_age_cache_equal_group_size/`
パターンで既に対象)。`pkill`や停電で止まっても、**同じコマンドをもう一度
打てば**キャッシュ済みの時点をスキップして再開する。

```bash
ps aux | grep export_k_model_equal_group_size_circuits | grep -v grep
tail -40 logs/k_model_equal_group_size_circuits.log
ls scripts/output/coupling_markov_production/viewer/_age_cache_equal_group_size/ | wc -l   # 完了済み時点数(81で全完了)
```

### 13-4. 結果の確認・push

完了すると`k_model_equal_group_size_circuits.csv`が書かれる(列:age・group・
metric・shared46_group_size・shared46_value・human_matched_group_size・
human_matched_value)。軽量な集計CSVなのでリポジトリに入れる:

```bash
git add scripts/output/coupling_markov_production/viewer/k_model_equal_group_size_circuits.csv
git commit -m "K-model: equal-group-size (46) vs human-matched-size circuit/widest-path comparison"
git push
```

**この結果の中身(人数の違いが値にどれだけ効くか)は、まだ解釈しない。**

---

## 14. K_self × K_out平面の個体密度ヒートマップ(事前登録外、探索的)

`scripts/plot_k_model_density_heatmap.py`(新規)は、`scripts/
compute_k_model_type_thresholds.py`と同じ集計(seed×gen 1000以降の
全時点をプールし、頻度で重み付け)から、K_self × K_out平面
([0,1]×[0,1])の2次元ヒストグラムを描く。中央値の点線(プールした
データから計算し直したもの——`type_thresholds.json`があればそこに
記録済みの値と一致するはずだが、このスクリプトは独立に計算する)で
4象限に区切り、各象限にK-安定型/K-拒絶型/K-とらわれ型/K-恐れ型の
ラベルを添える。「中央値で無理やり分けた型であり、集団は連続した
一つの雲」の注記付き。本番の生census(Mac local)を直接読むので、
このMacで実行する。

```bash
venv/bin/python scripts/plot_k_model_density_heatmap.py \
  --data-dir scripts/output/coupling_markov_production --seeds 70 --seed-offset 200 \
  --min-generation 1000 --bins 60 \
  --out-path scripts/output/figures/k_model_density_heatmap.png
```

(`--min-generation 1000`で、確証的分析・しきい値計算と同じgen 1000以降の
全時点[プローブ間隔50、81時点]をプールする。`--bins 60`は60×60マスの
解像度——Jaccardの図と違って1時点だけなので、計算はすぐ終わる
[数秒〜数十秒程度]。)

リポジトリに入れる:

```bash
git add scripts/output/figures/k_model_density_heatmap.png
git commit -m "K-model: K_self x K_out density heatmap"
git push
```
