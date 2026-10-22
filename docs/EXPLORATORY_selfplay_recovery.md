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
