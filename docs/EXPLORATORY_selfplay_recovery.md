# 探索的調査:進化した個体の「底からの回復」はなぜTFTに対して起きないのか

- 状態:**本番データに基づく結果確定**
- 対象ブランチ:`claude/markov-production-analysis`
- 位置づけ:**探索的・事前登録外**。`docs/PREREGISTRATION_markov_structure_evolution.md`の
  確証的仮説(H1〜H3)とは無関係であり、本報告の結果はH1〜H3の判定に一切影響しない。
- 使用コード:`evopersonality_abm/probe.py`(`run_selfplay_duel`・
  `recovery_time_from_forced_bottom`)、`scripts/export_markov_selfplay_recovery.py`。

## 1. 目的

`docs/MARKOV_STRUCTURE_EVOLUTION_RESULTS.md`の確証的結果(H1: TFT相手)が示すとおり、進化した
個体はTFT相手の3-gram網で底底底(相互裏切り)から和和和(相互協調)へほとんど戻れない。これが

- (a) 底からの回復そのものを行う仕組みを個体が持っていない(回復力の欠如)のか、
- (b) 回復の仕組み自体は持っているが、それが**仲間どうし(同じ進化的背景を持つ相手)でしか
  効かない**種類のもの(WSLS=win-stay-lose-shiftのような、相手が同じ規則を使っていないと
  機能しない互恵戦略)であるために、TFTという「仲間ではない」固定戦略に対しては発動しない
  のか、

を見分けるための探索的プローブである。

## 2. 方法

### 2.1 追加する3つの相手(人の3-gram網と同じ記号づけ)

既存の主セット(TFT・AllC・AllD・Random・Gradual)・拡張セット(GradualAppeasement・WSLS)に、
焦点の個体から見た相手として以下の3つを追加する。語彙は64語(Random・Gradualと同じ、無制限)。

| 相手 | 定義 | 実装 |
|---|---|---|
| **SELF** | 同じ遺伝子型どうし(両者とも実行ノイズ同一)。TFTのような固定戦略ではなく、**焦点の個体の進化した遺伝子型自身**を2つの独立したAgentで対戦させる。 | `probe.run_selfplay_duel(table, table, rng, ...)` |
| **POP** | 同じseed・同じ世代の集団から、頻度に比例して選んだ個体(30反復ごとに相手を引き直す)。 | `probe.run_selfplay_duel(table, partner_table, rng, ...)`(`partner_table`は`scripts/export_markov_selfplay_recovery.py`の`draw_partner_table`で頻度重み付き抽出) |
| **WSLS** | 拡張セットで既にプローブ済みの記録(`probe.OPPONENTS["WSLS"]`、固定のpavlov_table())をそのまま再利用。新規の対戦は行わない。 | 既存の本番/再プローブのmovesデータ(`*_moves.csv.gz`)から読み込み |

いずれも20反復(main_analysis_rounds)を分析対象とし、30反復(probe_repeats)、50世代おき
(`--probe-interval 50`。重い場合は250世代おきに切り替え可能)。

### 2.2 主要指標(人の3-gram網と同じ形式)

- 底底底→和和和、和和和→底底底のwidest path(`markov_apply.bootstrap_aggregate_with_sampler`、
  既存コードをそのまま再利用、人データとの比較はしない=`human_weights_by_group={}`)。
- 各ノード(3-gram)の出現率(`export_markov_edge_weights_for_viewer.node_appearance_rates_over_draws`、
  既存コードをそのまま再利用)。
- 系統間抽出(cross_lineage、56人抽出)を200回。
- 出力は既存のビューア用JSONと**同じ形式**(`scripts/output/figures/markov_selfplay_edges.json`・
  `markov_selfplay_nodes.json`、トップレベルキーが`"SELF"`・`"POP"`・`"WSLS"`になるだけで、
  既存の`markov_edge_weights.json`・`markov_node_freq.json`は一切変更しない)。

### 2.3 補助指標:「底からの回復時間」(人の物差しの外であることを明記)

**この指標は人の実験のどの測定とも対応しない、この調査のために新しく作ったものである。**
人の3-gram網・widest pathはいずれも実際に対戦記録に現れたパターンの構造を見るものだが、
「最初の3回を強制的にDD(底底底)に固定したあと、何手目で初めて2回連続のCC(和和和)に
戻るか」という問いは、人の実験では一度も測定されていない操作的な介入実験である。

- 最初の3回を強制的にDD(相互裏切り)に固定する。
- そのあと20回(main_analysis_roundsと同じ`--main-analysis-rounds`)、焦点の個体を
  TFT・SELF・POP・WSLSの相手と自由に対戦させる(`probe.recovery_time_from_forced_bottom`)。
- 2回連続のCC(和和和)が最初に現れた手番(1始まり)を「回復時間」として記録する。
  20回以内に一度も現れなければ「回復しなかった」(右側打ち切り)として扱う。
- 世代ごとに、相手別の**中央値の回復時間**と**20回以内に回復しなかった割合**を報告する
  (`export_markov_selfplay_recovery.summarize_recovery_times`)。
- 出力:`scripts/output/figures/markov_selfplay_recovery_time.csv`
  (列:`opponent, age, n, n_recovered, median_recovery_round, never_recovered_rate`)。

### 2.4 乱数の流れ

既存のmarkov_sweepパイプラインと同じ慣行(`random.Random(f"{master_seed}_{seed}_{age}")`、
世代ごとにプロセスプールで並列化しても再現可能)。1つの(seed, age)の中で、SELF/POPの
series構築 → 底からの回復時間の計算、という順でこの同じrngを消費する
(`build_pools_for_age`が両方をこの順で計算する)。

## 3. 単体テスト(`tests/test_probe.py`・`tests/test_export_markov_selfplay_recovery.py`)

依頼どおり、以下の2つを含む(いずれもpavlov_table()のwin-stay-lose-shift規則を手計算で
検証した上でコード化):

1. **`test_recovery_time_wsls_vs_tft_never_recovers`/`..._cycles_through_bottom_sucker_temptation`**:
   WSLSが強制的な底底底から始めてTFTと対戦すると、惨(C,D)→搾(D,C)→底(D,D)の3手サイクルを
   永遠に繰り返し、20回以内に2連続CCへ戻らない(`recovery_time_from_forced_bottom`が`None`を
   返す)。TFTは相手の直前の手を鏡写しするだけで自ら協調を**開始**しないため、両者とも
   「相手の最後の手がD」という理由で裏切りを続け、WSLS側だけが「1回報復→次は協調に戻る」を
   試みても、TFTが即座にそれを裏切りと誤認して報復し直す構造的な非対称性が原因。
2. **`test_recovery_time_wsls_vs_wsls_recovers_two_rounds_after_a_single_error`**:
   WSLS対WSLSでは、安定した相互協調の最中に1回だけ誤り(片側が1回だけ裏切る)を注入すると、
   その1手後にいったん両者とも裏切り(WSLSの「lose-shift」が両側で同時に発動)、
   その2手後には相互協調に回復し、以後安定する。

(余談・探索的に発見:`test_recovery_times_for_seed_age_self_mutual_tft_never_recovers`で
確認したとおり、**TFTどうしの対戦もまた、強制的な底底底からは自発的に回復しない**
——TFTは「相手の直前の手を鏡写しする」だけで自ら協調を開始する仕組みを持たないため、
両者が同じTFTでも鏡写しの無限ループに陥る。これはSELF側の相手がTFT型の遺伝子型である
場合の重要な対照例になる。)

全体のテストスイート(`python3 -m pytest -q`)で403件すべて合格を確認済み(パイロット規模の
スモークテストも別途、`scripts/output/markov_sweep_pilot_noise00`に対して実行し、
4つの出力ファイルすべてが期待どおりの形式で書き出されることを確認済み)。

## 4. Mac本番実行

手順・所要時間の見積もりは`docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md`12節を参照。

## 5. 結果

データ:`markov_sweep_production_noise01`(主条件、seed 100〜169、70/70完走)、gen 1000〜5000
(50世代おき、81時点)、cross_lineage抽出(56人、200回)。コミット`f7422aa4`。

### 5.1 Widest path(底↔和の到達可能性、81時点平均)

| 相手 | rate_bottom3_present(底底底が現れる率) | rate_top3_present(和和和が現れる率) | rate_recovery_undefined(底→和が未定義の率) | mean_recovery(定義された回のみ) | mean_descent(定義された回のみ) |
|---|---|---|---|---|---|
| TFT(主条件・確証的結果、scheme B/group_size 56) | **1.000** | (常に高い) | **0.034** | 0.125 | 0.156 |
| SELF | 1.000 | 1.000 | 0.0001 | 0.173 | 0.153 |
| POP | 1.000 | 1.000 | 0.0002 | 0.179 | 0.160 |
| WSLS | **0.016** | 1.000 | **0.986** | 0.092(71/81時点のみ定義) | 0.108 |

(TFTの行は`docs/MARKOV_STRUCTURE_EVOLUTION_RESULTS.md`の確証的結果と同じ本番データ・同じ
scheme[cross_lineage, group_size=56]から、このセクションのための参考として再集計した値。
確証的判定そのものは変更しない。)

TFTとWSLSはどちらも`rate_recovery_undefined`が高いという意味では似ているように見えるが、
**原因が正反対である**:

- **TFT**: 底底底は常に現れる(`rate_bottom3_present=1.000`)。未定義率は低い(0.034)——
  つまりTFT相手では個体は頻繁に底に落ちるし、底→和のwidest pathはほとんどの回で**計算できる**。
  定義できた回の`mean_recovery`(0.125)がH1の確証的結果(r(S)がr(M)・r(F)より有意に低い)で
  示されたとおり低い値にとどまる——**底には落ちるが、そこから這い上がる経路が構造的に弱い**。
- **WSLS**: 底底底がほとんど現れない(`rate_bottom3_present=0.016`、81時点中71時点でしか
  recoveryが1回も定義されない)。未定義率が高い(0.986)のは、**底→和の経路が弱いからではなく、
  そもそも底に落ちること自体がほとんど起きないから**(次節で確認)。

### 5.2 確認:WSLSの`rate_recovery_undefined≈.98`は底底底ノードが現れないことが原因か

**はい、その通り。** `markov_selfplay_nodes.json`(各3-gramの出現率、200回の系統間抽出平均)で
確認すると、WSLS相手の底底底(3連続相互裏切り)の出現率は81時点すべてで**ほぼゼロ**:

| 世代 | 底底底の出現率(WSLS相手) | 和和和の出現率(WSLS相手) |
|---|---|---|
| 1000 | 0.0001 | 0.4729 |
| 2000 | 0.0004 | 0.4153 |
| 3000 | 0.0004 | 0.4655 |
| 4000 | 0.0001 | 0.5001 |
| 5000 | 0.0004 | 0.5438 |

(参考:SELF/POPの底底底の出現率は同じ世代で0.33〜0.44と、WSLSの1000倍以上ある。)

すなわち、仮想参加者20人中、WSLS相手に底底底を一度でも通る人は**200回の抽出のどれを見ても
事実上ゼロ**(出現率0.0001〜0.0004は20人中0〜1人未満に相当)。`rate_bottom3_present`
(抽出56人のネットワークに底底底ノード自体が現れる率)が0.016しかないのも、個々の出現率が
ほぼゼロであることの直接の結果であり、widest_path計算の側のバグではない。

**解釈**:WSLS(win-stay-lose-shift)は実行ノイズ0.01程度の誤りからはほぼ即座に協調へ戻る
ため、自然な20手の対戦の中で「3連続相互裏切り」という深い底にまで落ち込むことがほとんど
起きない。TFTのように一度の裏切りを延々と報復応酬し合って底に留まり続けることがない
——つまりWSLSは「底に落ちない」のであって、「底から戻れない」わけではない。これは
2.3節の単体テスト(WSLS対WSLS:1回の誤りのあと2手で和に戻る)が示す回復の速さとも整合する。
この点はTFT(底には落ちるが這い上がれない)と構造的に異なり、widest pathの指標だけを見て
両者を同じ「回復が効かない」カテゴリに入れるのは誤りである。

### 5.3 補助指標:底からの回復時間(強制介入、人の物差しの外)

20回の自由対戦に入る前に最初の3回を強制的に底底底に固定した上での回復時間(2.3節参照)。

| 相手 | never_recovered_rate(20回以内に回復しない率、81時点平均) | median_recovery_round(回復した回のみ、81時点平均) |
|---|---|---|
| TFT | **0.858** | 4.44 |
| SELF | 0.523 | 2.0 |
| POP | 0.540 | 2.0 |
| WSLS | 0.502 | 2.0 |

強制的に底に落とした場合、SELF・POP・WSLSはいずれもTFTより明確に速く・高い確率で回復する
(回復するときはほぼ全て2手後、未回復率もTFTの約60%)。WSLSは5.1〜5.2節で見たとおり
「自然には底に落ちない」が、**底に落とされればTFTよりずっと速く這い上がる**——つまりWSLSは
回復の仕組みそのものを持っており、それは同じ遺伝子型どうし(SELF)や同じ集団の個体(POP)に
限らず、WSLSという全く別の(進化していない)固定戦略に対しても機能する。

(ただし、SELF/POP/WSLSでも未回復率は50%程度あり、「仲間」相手でも常に回復するわけではない
——回復力は存在するが万能ではない。)

### 5.4 まとめ:(a)回復力の欠如 か (b)仲間限定の回復力 か

結果は(a)でも単純な(b)でもなく、**TFTという相手の構造的な特殊性**を指し示す:

1. 回復の仕組み自体は存在する(SELF・POP・WSLSいずれに対しても、強制的に底から回復できる
   ——5.3節)。したがって(a)「回復力の欠如」は支持されない。
2. その仕組みは同じ遺伝子型(SELF)・同じ集団(POP)に限らず、**進化していない固定戦略
   (WSLS)**に対しても機能する。したがって(b)を「同じ進化的背景を持つ仲間限定」と
   狭く解釈する説明も支持されない。
3. TFTだけが例外的に回復を許さない。2.3節のWSLS対TFTの単体テストで確認したとおり、
   TFTは相手の直前の手を鏡写しするだけで自ら協調を**開始**しない。底に落ちた状態
   (両者ともD)から、どちらかが先に協調を試みても、TFTはその1手だけでは状況を変えず
   (自分の直前の手がDのままなので次もDを返す)、相手が協調を続けて初めて協調に転じる
   ——この「先に動かない」性質が、進化した個体がどんな回復戦略を持っていてもTFT相手には
   発動しにくくする(5.1節でTFT相手のmean_recoveryが低いこと、5.3節でTFT相手の未回復率が
   他の3相手の約1.6〜1.7倍高いことの両方と整合する)。

## 6. 記憶の深さ別の回復時間、およびWSLS規則を持つ記憶深さ1個体の割合(探索的・事前登録外)

5.3節の「底からの回復時間」を、ビューア(10節・11節)で使っているのと同じ3つの実効記憶深さ群
(depthLow/depthMid/depthHigh)に分けて世代ごとに見る。あわせて、各世代で記憶の深さ1(名目上の
`memory_depth==1`)の個体のうち、規則そのものがWSLSの win-stay-lose-shift(`pavlov_table()`の
テーブル`[1,0,0,1]`=「前回が和か搾なら同じ手、惨か底なら手を変える」)と一致する個体の頻度
加重割合を求める。

### 6.1 方法

使用コード:`scripts/export_markov_selfplay_recovery_by_depth.py`(新規)。

- **回復時間の深さ別集計**:`probe.recovery_time_from_forced_bottom`を、既存の
  5.3節と**完全に同じ定義**(最初の3回を底底底に固定→20回自由対戦→2連続CCの最初の手番、
  相手はTFT・SELF・POP・WSLS)で再計算するが、今回は焦点の個体の**実効記憶深さ**
  (`evopersonality_abm.lindgren.effective_memory_depth`)で結果をdepthLow/depthMid/depthHigh
  に分けて集計する(`export_markov_edge_weights_for_viewer.depth_ranges`をそのまま再利用、
  **ビューアと同じ`--depth-cutpoints`**を使うこと)。POPの相手は深さ群で絞らず、常に
  **集団全体**から頻度加重で選ぶ(焦点の個体側の深さだけで結果を分類する)。
- **なぜ既存のキャッシュをそのまま読めないか**:`export_markov_selfplay_recovery.py`の
  `_age_cache_selfplay/`は、`process_one_age`が`summarize_recovery_times`で個体ごとの
  回復時間を**世代内で即座にプールしてしまう**ため、個体の深さを後から取り出せる形では
  保存されていない。深さ別に分けるには同じ`census`から回復時間を**再計算**する必要がある
  ——ただし**新たな進化は一切行わず**、既存の本番census(`seed_X_probe_age_G_census.csv.gz`、
  8〜9節で確定済み)を読むだけで、SELF/POP用の100手の系列構築やwidest path・辺の重み・
  ノード出現率のブートストラップ(5節の処理時間の大部分を占めていた部分)は一切行わない
  ため、**5節のスクリプトより大幅に軽い**(6.2節参照)。
- **記憶深さ1・WSLS規則の割合**:対戦シミュレーションを一切行わない、census記録のみの
  集計。各(seed, age)のcensusで`table.memory_depth==1`(名目上の深さ1、2^(2×1)=4要素の
  ルックアップテーブル)の個体を数え、そのうち`table.table == (1,0,0,1)`(prehistoryは
  問わない——「同じ規則」は4通りの場面すべてで同じ手を返すことを指し、対戦開始前の
  仮定履歴の違いは問わない)と完全一致する個体の頻度を数える。全70 seedをプールした
  頻度加重割合として世代ごとに報告する。
  (注:これは`effective_memory_depth`に基づくビューアの3群とは**別の基準**であり、
  `duplicate_memory`で深さ2以上に増えた「実効的にはWSLSと同じ」個体はこの集計には
  含まれない——名目上のテーブルが4要素のままWSLSと一致する個体だけを数える。)

### 6.2 単体テスト・スモークテスト

`tests/test_export_markov_selfplay_recovery_by_depth.py`(15件、全合格)。全体テストスイート
(`python3 -m pytest -q`)418件合格を確認済み。パイロットデータ(`markov_sweep_pilot_noise00`、
10 seed)で本番と同じ`--probe-repeats 30 --free-rounds 20`による実行を確認し、1(seed,age)
単位あたり約0.31秒(5節のSELF/POP系列構築・widest path集計を含むスクリプトの約1/10の
コスト——このスクリプトは対戦シミュレーション自体は同程度の回数行うが、100手のSELF/POP系列
構築や200回のブートストラップ集計を一切行わないため)。キャッシュの再開(同じコマンドを
2回実行して2回目が即座に完了すること)も確認済み。

### 6.3 Mac本番実行

手順・所要時間の見積もりは`docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md`13節を参照。

### 6.4 結果

データ:`markov_selfplay_recovery_time_by_depth.csv`(972行、81時点×4相手×3群、欠損なし)・
`markov_depth1_wsls_rule_fraction.csv`(81行)。コミット`c3714880`。集計・分解には
`scripts/analyze_selfplay_recovery_by_depth.py`(新規、テスト8件)を使用。

**集計の単位についての注記**:ここでの「群の構成比」は、集団内の各遺伝子型の頻度では
なく、**その世代・その群に属する"異なる遺伝子型"の数**(70 seed分をプールしたもの)に
基づく——この探索的プローブの他のすべての集計(5節・12節)と同じ、1遺伝子型=1個体
として扱う慣行(`probe.census_genotypes`の重複排除)に合わせたもの。

#### 6.4.1 深さ群別の回復時間(表:81時点の平均)

| 相手 | depthLow | depthMid | depthHigh |
|---|---|---|---|
| TFT(never_recovered_rate) | 0.880 | 0.864 | 0.853 |
| SELF | 0.731 | 0.576 | 0.453 |
| POP | 0.783 | 0.594 | 0.459 |
| WSLS | 0.685 | 0.534 | 0.444 |

(値は`mean_rate_by_group`、81時点の単純平均。5.3節の深さを分けない集計(TFT全体
never_recovered_rate=0.858、SELF/POP/WSLS全体0.50〜0.54)は、この3群の加重平均と
一致する。)

**SELF・POP・WSLSでは、深さが深い群ほど回復しやすい**(depthLowの0.68〜0.78から
depthHighの0.44〜0.46まで、はっきりした差)。**TFT相手では、この差はごくわずか**
(0.880→0.853、3群の違いは0.03未満)——どの深さ群でもTFT相手にはほとんど回復しない、
という5節の結論は深さ群を分けても変わらない。

#### 6.4.2 集団全体の回復の伸びの分解(within-group と composition)

事前登録外の記述的分解であることを明記する:各世代の集団全体の回復率(=1-never_recovered_rate、
3群の加重平均)の、gen 1000→gen 5000の変化を、(a)**各群自身の回復率の変化**(within-group、
構成比を固定した場合の伸び)と、(b)**3群の構成比の変化**(composition、各群自身の回復率を
固定した場合の伸び)に、残差なく厳密に分解した(`analyze_selfplay_recovery_by_depth.decompose`、
連続する2時点ごとに中点法で分解し81時点分を合算——経路が単調でなくても厳密に合計と一致する)。

| 相手 | 回復率の変化(gen1000→5000) | within-group寄与 | composition寄与 |
|---|---|---|---|
| TFT | 0.134→0.170(+0.037) | +0.040(108%) | -0.003(-8%) |
| SELF | 0.339→0.559(+0.220) | +0.183(83%) | +0.037(17%) |
| POP | 0.327→0.534(+0.207) | +0.162(79%) | +0.044(21%) |
| WSLS | 0.402→0.577(+0.175) | +0.141(81%) | +0.019(19%) |

SELF・POP・WSLSのいずれも、集団全体の回復率の伸びの**大部分(約79〜83%)は各群自身の
回復率が世代とともに上がっていくこと(within-group)**によるもので、構成比の変化
(どの深さ群が集団に多いか)の寄与は**17〜21%と小さい**。すなわち、集団全体で回復が
進む主因は「回復しやすい深い群に個体が集まっていくこと」ではなく、「同じ深さ群の中でも
個体がより回復しやすい戦略へ進化していくこと」である。TFT相手では全体の伸びがそもそも
小さく(+0.037)、within-group寄与がほぼ全部(108%、composition寄与はわずかに負)——
TFT相手の回復しにくさは、深さ構成の変化ではほとんど説明できない。

**相関であって因果ではないことの明記**:6.4.1節の「深さが深いほど回復しやすい」という
関係は、**進化した個体間の相関**であり、記憶の深さが回復力を**引き起こしている**ことを
主張するものではない。記憶の深さは進化の過程で他の形質(例:ルックアップテーブルの
どの固定点に落ち着くか、WSLS型の規則をどれだけ正確に実装しているか)とも同時に変化して
おり、深さそのものが原因なのか、深さと一緒に進化した別の形質が回復力の実際の原因なのかは、
この記述的プローブだけでは切り分けられない。6.4.2節のwithin/composition分解も、あくまで
「集団全体の伸びのうち、群の構成比の変化で説明できる部分」と「群の中での変化で説明できる
部分」を会計的に分けたものであり、どちらの部分にも深さ以外の交絡要因が入りうる。

#### 6.4.3 記憶深さ1個体中のWSLS規則一致率

81時点すべてで値が得られた(欠損なし)。全体の平均は4.5%(最小0.4%〜最大12.1%)、
前半20時点(gen 1000〜1950)平均4.0%に対し後半20時点(gen 4050〜5000)平均5.6%と、
**弱い増加傾向はあるが、世代間のばらつきが大きく、強い傾向とは言えない**(単純な線形
回帰の傾きは、4000世代全体で+1.6ポイント程度)。

あわせて、記憶深さ1の個体数そのものが世代とともに減少している(n_depth1:gen 1000で
597→gen 5000で221、約1/3に減少)——集団全体でより深い記憶を持つ個体が増える
(6.4.1節のdepthHigh群の構成比の増加、および`scripts/output/50seed_confirmation/
memory_depth_evolution_50seed.png`で既に確認されている傾向)のに伴い、記憶深さ1の
個体という母数自体が縮小していく。したがって後半のWSLS規則一致率の値は、より小さい
母数に基づく推定であり、不確実性が大きい点に注意。

### 6.5 学会スライド用の図(初版、保持)

`scripts/output/figures/markov_selfplay_recovery_self_vs_tft_by_depth.png`
(`scripts/analyze_selfplay_recovery_by_depth.py`の`plot_self_vs_tft`が生成、**以降の改訂でも
このファイル・この関数は変更していない**)。SELF相手のnever_recovered_rateを深さ3群
(depthLow=青・depthMid=橙・depthHigh=赤、各世代の値をそのまま線でつないだもの、平滑化なし)
で分け、比較のためTFT相手の深さを分けない全体値を薄い破線で重ねた。6.4.1節の「SELF相手
では深さが深いほど回復しやすい」「TFT相手ではどの深さでもほぼ回復しない」の両方が視覚的
に確認できる。

### 6.6 学会スライド用の図(改訂版:回復率・ブートストラップ帯・ヒラギノ角ゴ)

6.5節の図への4つの改訂依頼を反映した版。**元の図(6.5節)は変更していない**——別ファイル名
(`markov_selfplay_recovery_rate_by_depth_academic_{SELF,POP,WSLS}.png`)で追加した。

1. **縦軸を回復率にした**:never_recovered_rateではなく「20回のうちに和和和へ戻れた割合」
   (1-never_recovered_rate)を縦軸にする。
2. **seedをまたいだブートストラップの95%帯**:各線(3つの深さ群+TFT参考線)に薄い帯を
   付けた。70 seedを復元抽出で2000回選び直し、そのたびに再プールした回復率を計算する
   ——本プロジェクトの他の箇所(確証的分析のseedブートストラップ)と同じ「seed単位で
   まるごと選び直す」方式。既存の集計CSV(`markov_selfplay_recovery_time_by_depth.csv`)は
   seedごとの内訳をプールして捨ててしまっているため、`export_markov_selfplay_recovery_by_depth.py`
   を拡張し、新しく`markov_selfplay_recovery_time_by_depth_per_seed.csv`(seed・世代・相手・
   深さ群ごとのn・n_recovered)を追加で書き出すようにした(既存の2つの出力ファイルの値は
   変えていない——同じmaster_seed・同じ乱数の流れからの再計算で、seedごとの内訳を"ついでに"
   記録するようにしただけ)。TFT参考線(深さを分けない全体値)の帯は、3つの深さ群のseedごとの
   カウントをseed内で合算して作る(`tft_pooled_per_seed`)——別途TFT専用の集計をやり直す
   必要はない。
3. **フォント**:ヒラギノ角ゴ(`Hiragino Sans`等)を使う。このクラウド環境にはヒラギノが
   インストールされていないため(`fc-list`で確認済み)、スクリプトは実行時にインストール
   済みフォントを自動確認し、見つからなければ`IPAGothic`に切り替えて警告を出す
   (`select_japanese_font`、テスト済み)。**Macで実行すればヒラギノ角ゴが自動で使われる**
   ——Mac用の手順は`docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md`14節を参照。
4. **凡例の文言**:depthHigh→「記憶が深い」、depthMid→「中くらい」、depthLow→「浅い」、
   TFT参考線→「TFT相手(参考)」に変更した(この順で表示)。
5. **POP・WSLSの図も追加**:SELFと同じ構成(3つの深さ群+TFT参考線)で、POP相手・WSLS相手
   それぞれの図を作った。

#### 結果

**(Macでの本番実行待ち。`scripts/output/figures/markov_selfplay_recovery_time_by_depth_per_seed.csv`
が本番データでpushされ、Mac上でこのスクリプトを実行してヒラギノ角ゴ版の図が得られたら、
このセクションに画像の要点を追記する。)** コードはテスト18件(新規10件)・単体テスト・
パイロットデータでの一気通貫のスモークテスト(10 seed・3時点、ブートストラップ帯を含む
4枚のPNGが正しく書き出されることを確認——クラウドではヒラギノがないためIPAGothicに
自動フォールバックし、警告が出ることも確認済み)で検証済み。
