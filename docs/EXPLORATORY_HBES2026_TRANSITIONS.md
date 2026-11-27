# HBES2026前の探索的分析:行動遷移構造(3指標・交代と報復・記憶の深さ別の網)

**この文書が記録する分析は、事前登録(`docs/PREREGISTRATION_k_model_markov.md`)に
含まれない、探索的な分析である。** 確証的な判定(支持/支持されず)はここでは行わない。
数値と手順の記録のみを目的とする。

## 0. 共通の設定

- データ: `scripts/output/coupling_markov_production`(seed 200〜269の70本、
  世代1000〜5000)。Mac local(gitignore対象)のため、このリポジトリには
  集計済みの結果(CSV・JSON)のみを収めている。
- 類型の4分割: `scripts/output/coupling_markov_production/analysis/type_thresholds.json`
  が記録する、K_self・K_outそれぞれの中央値分割しきい値を使う
  (`scripts/compute_k_model_type_thresholds.py`で計算済み)。
- 3つのスクリプトとも、頻度(`frequency`列)で重みづけた集計を行い、
  seed単位のブートストラップで95%範囲を出す(`n-boot`回の復元抽出)。
- 以下の「人のデータとの比較」の数値は、安念から提供された集計値をそのまま記載した
  もので、個人の記録(質問紙の得点・個人の対戦系列)はこのリポジトリに含まれない。
  群の分け方: 見捨てられ不安と親密性回避のz得点がともに0より大きい人がFearful
  (69人)、ともに0以下がSecure(56人)、それ以外がMix(46人)。この分け方で
  `docs/human_network_weights.csv`の網の値が再現されることを確認済み。

---

## 1. `scripts/exploratory_hbes2026/k_agg.py` — 協力率・入口・到達

### 目的

Kモデルの対戦記録(moves)から、人の実験と同じ3つの指標を、類型(K-安定型/
K-拒絶型/K-とらわれ型/K-恐れ型)×相手(AllC/AllD/Gradual/Random/TFT)ごとに
集計する。最初の20回だけを使う(人の実験と同じ)。

- **協力率**(`coop_rate`): 20回のうち自分が協力(和または惨の状態)した割合。
- **入口**(`entry_C_after_bottom`): 底(相互裏切り)の直後に協力した割合。
- **到達**(`reach_CCC_after_first_bottom`): 最初の底のあと、和和和(3連続の
  相互協調)まで行けたかどうか。

### 実行コマンド

```bash
venv/bin/python scripts/exploratory_hbes2026/k_agg.py \
  --age-start 1000 --age-end 5000 --age-step 50 \
  --workers 5 --n-boot 2000 \
  --out docs/exploratory_hbes2026_results/k_by_type_opponent.csv
```

(`--data-dir`・`--thresholds-json`は省略時、本番の既定
`scripts/output/coupling_markov_production`・その`analysis/type_thresholds.json`
を使う。)

### 出力ファイル

`docs/exploratory_hbes2026_results/k_by_type_opponent.csv`
(列: `opponent,metric,group,mean,n_seeds,ci_lo,ci_hi`。`n_seeds=70`。)

### 主な結果(協力率、K-安定型/K-恐れ型、人との比較)

| 相手 | K-安定型 | K-恐れ型 | 人Secure | 人Fearful | 人での判定(安念提供) |
|---|---|---|---|---|---|
| AllC | 0.836 | 0.800 | .42 | .52 | 差があると言える |
| AllD | 0.393 | 0.369 | .42 | .41 | 差があるとは言えない |
| GRA  | 0.696 | 0.621 | .47 | .48 | 差があるとは言えない |
| RAN  | 0.558 | 0.537 | .42 | .49 | 差があるとは言えない |
| TFT  | 0.747 | 0.693 | .45 | .49 | 差があるとは言えない |

### 主な結果(入口、K-安定型/K-恐れ型、人との比較)

| 相手 | K-安定型 | K-恐れ型 |
|---|---|---|
| AllD | 0.552 | 0.424 |
| GRA  | 0.571 | 0.440 |
| RAN  | 0.541 | 0.438 |
| TFT  | 0.545 | 0.552 |

人側は「どの相手でも差は.04以内で、差があるとは言えない」(安念提供)。

### 主な結果(到達、K-安定型/K-恐れ型、人との比較)

| 相手 | K-安定型 | K-恐れ型 | 人Secure | 人Fearful |
|---|---|---|---|---|
| AllD | 0.000 | 0.000 | ー | ー |
| GRA  | 0.018 | 0.045 | .02 | .00 |
| RAN  | 0.306 | 0.291 | .08 | .15 |
| TFT  | 0.238 | 0.386 | .29 | .29 |

(AllC相手は「底」状態が原理的に生じないため、入口・到達は計算されない。)

全48行(全4類型×3指標×5相手、ブートストラップ95%範囲を含む)は
`docs/exploratory_hbes2026_results/k_by_type_opponent.csv`を参照。

---

## 2. `scripts/exploratory_hbes2026/k_alt.py` — 交代・報復・回帰

### 目的

相手TFT・Gradual・Randomとの対戦で、「交代」(搾⇄惨の入れ替わり)の指標と、
171人全員を使った回帰を計算する。

- **交代の回数**(`alt_count`): 20回のうち搾と惨が入れ替わった回数。
- **交代3連**(`alt3_any`): 搾惨搾または惨搾惨という3連が一度でもあったか。
- **報復率**(`retaliation`): 惨(自分C・相手D)の直後に自分が裏切る割合。
- **交代の最長連続**(`alt_longest`): 交代が連続した最長の長さ。
- **回帰**: 交代の回数・報復率を、不安相当(= −z(K_self))・回避相当
  (= −z(K_out))・その積に、頻度重みつき最小二乗で回帰(seedごとに世代平均し、
  seedのブートストラップで95%範囲)。

### 実行コマンド

```bash
venv/bin/python scripts/exploratory_hbes2026/k_alt.py \
  --age-start 1000 --age-end 5000 --age-step 50 \
  --workers 5 --n-boot 2000 \
  --out-prefix docs/exploratory_hbes2026_results/k_alt
```

### 出力ファイル

- `docs/exploratory_hbes2026_results/k_alt_by_type.csv`
  (類型ごとの平均 + `diff_Secure_minus_Fearful`行。列: `opponent,metric,group,mean,n_seeds,ci_lo,ci_hi`)
- `docs/exploratory_hbes2026_results/k_alt_regression.csv`
  (列: `opponent,metric,term,b,n_seeds,ci_lo,ci_hi`。`term`は切片・不安相当(-K_self)・
  回避相当(-K_out)・積の4つ)

### 主な結果(交代の回数、K-安定型/K-恐れ型、人との比較)

| 相手 | K-安定型 | K-恐れ型 | 差(安定−恐れ) | 判定 | 人Secure | 人Fearful |
|---|---|---|---|---|---|---|
| GRA | 1.297 | 1.292 | +0.006 | 差があるとは言えない | ー | ー |
| RAN | 3.993 | 2.889 | +1.104 | 差があると言える | ー | ー |
| TFT | 6.281 | 4.102 | +2.179 | 差があると言える | 3.2 | 4.1 |

(「判定」は、いずれもseedのブートストラップの95%範囲による、探索的な判定。)

TFT相手について:**人はFearfulの方が交代が多い(3.2対4.1、差−0.9回、「ぎりぎり
差があると言える」と安念から提供)のに対し、KモデルはK-安定型の方が交代が多い
(6.28対4.10、差+2.18、差があると言える)**。向きが逆であることをそのまま記録する
(解釈はしない)。

### 主な結果(報復率、K-安定型/K-恐れ型、人との比較)

| 相手 | K-安定型 | K-恐れ型 | 差(安定−恐れ) | 判定 |
|---|---|---|---|---|
| GRA | 0.827 | 0.650 | +0.177 | 差があると言える |
| RAN | 0.878 | 0.679 | +0.199 | 差があると言える |
| TFT | 0.815 | 0.584 | +0.231 | 差があると言える |

(「判定」は、いずれもseedのブートストラップの95%範囲による、探索的な判定。)

TFT相手の人の報復率は Secure .48 / Fearful .56(差があるとは言えない、安念提供)。
Kモデルは3相手ともK-安定型の方が報復率が高く、差があると言える。人のTFTでの
向き(Fearfulの方が高い、ただし差があるとは言えない)とKモデルの向き(K-安定型
の方が高い)は逆である。

### 主な結果(回帰:不安相当・回避相当・積)

| 相手 | 指標 | 不安相当(-K_self) | 回避相当(-K_out) | 積 |
|---|---|---|---|---|
| GRA | alt_count | +0.132(差があると言える) | −0.109(差があると言える) | +0.012(差があると言える) |
| GRA | retaliation | −0.007(差があるとは言えない) | −0.091(差があると言える) | −0.012(差があると言える) |
| RAN | alt_count | +0.002(差があるとは言えない) | −0.577(差があると言える) | −0.022(差があると言える) |
| RAN | retaliation | −0.013(差があると言える) | −0.095(差があると言える) | −0.012(差があると言える) |
| TFT | alt_count | −0.059(差があるとは言えない) | −1.111(差があると言える) | −0.124(差があると言える) |
| TFT | retaliation | −0.021(差があると言える) | −0.109(差があると言える) | −0.012(差があると言える) |

(「差があると言える/差があるとは言えない」は、いずれもseedのブートストラップの
95%範囲による、探索的な判定(事前登録の確証的分析のような置換検定との二重の
確認はしていない)。Kモデルは1時点あたりの個体数が人より多いので、小さな係数
でも「差があると言える」判定になりやすい点に注意。)

人の171人全員の回帰(安念提供):TFT相手で見捨てられ不安が+0.57回/SD(p=.041、
交互作用は有意でない)、RAN相手で親密性回避が+0.37回/SD(p=.026)。比較の数が
多いので、どちらも探索的な結果として扱う、とのこと。

KモデルのTFT相手の不安相当の係数は−0.059で差があるとは言えない(人の+0.57とは
向きも大きさも対応しない)。KモデルのRAN相手の回避相当の係数は−0.577で差が
あると言える(人の+0.37とは向きが逆)。

全データ(切片を含む全24行)は`docs/exploratory_hbes2026_results/k_alt_regression.csv`
を参照。

---

## 3. `scripts/exploratory_hbes2026/k_depth_viewer_all.py` — 記憶の深さ別の網

### 目的

世代1000〜5000(100おき)の個体を、実効の記憶深さ(0〜7、7は7以上をまとめる)
×類型×相手8種(TFT・AllC・AllD・Random・Gradual・WSLS・SELF・POP)でまとめ、
人と同じ人数の群を系統間抽出(別々の系統から1体ずつ、100回抽出の平均)して、
平均Jaccardの網と語の出現率を、深さごとに作る。

### 実行コマンド

```bash
venv/bin/python scripts/exploratory_hbes2026/k_depth_viewer_all.py \
  --age-start 1000 --age-end 5000 --age-step 100 \
  --workers 5 --reservoir 40 --depth-cap 7 --n-draws 100 --min-group 15 \
  --out-dir docs/exploratory_hbes2026_results/k_depth_all
```

### 出力ファイル

- `docs/exploratory_hbes2026_results/k_depth_all/k_depth_edges.json`
  (`{"abm": {相手: {類型/Mix: {深さ: [[語,語,重み], ...]}}}, "meta": {...}}`)
- `docs/exploratory_hbes2026_results/k_depth_all/k_depth_nodes.json`
  (同じ構造で、辺の代わりに語ごとの出現率)
- `docs/exploratory_hbes2026_results/k_depth_all/k_depth_group_size.csv`
  (列: `grouping,depth,seeds_available,group_size,individuals`。深さごとの
  利用可能系統数・実際に使った群の人数・該当個体の総数)

### 主な結果(群の人数、TFT相手時点)

| 類型/Mix | 目標人数 | 深さ0 | 深さ3 | 深さ7 |
|---|---|---|---|---|
| K-安定型 | 56 | 56(系統69) | 56(系統70) | 56(系統70) |
| K-拒絶型 | 46 | 46(系統67) | 46(系統70) | 46(系統70) |
| K-とらわれ型 | 46 | 46(系統68) | 46(系統70) | 46(系統70) |
| K-恐れ型 | 69 | 69(系統69) | 69(系統70) | 69(系統70) |
| 合成Mix | 46 | 46(系統69) | 46(系統70) | 46(系統70) |

いずれの深さでも目標人数(人の群に合わせた人数)に達しており、
`--min-group 15`の足切りにかかる深さ・類型はない。全32行
(深さ0〜7×5グループ)は`k_depth_group_size.csv`を参照。

### 主な結果(網の形、安念による読み取り)

固定戦略の相手でも仲間(SELF)相手でも、網が大きく変わるのは深さ0→1の間だけで、
類型の差はどの深さでもほぼ同じだった。以前の「深い記憶ほどK-恐れ型が戻れる」
(強制的に底から始めた回復の割合、`docs/K_MODEL_MARKOV_RESULTS.md`)は、
網の太さ(Jaccardの重み)には表れなかった。

---

## 4. ビューアとの関係

`k_depth_viewer_all.py`が出力する`k_depth_edges.json`/`k_depth_nodes.json`は、
claude.ai上のビューア(人とKモデルの網を並べる版)に、「Kモデルの軸=記憶の深さ」
として既に組み込まれている。
