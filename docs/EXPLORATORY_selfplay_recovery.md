# 探索的調査:進化した個体の「底からの回復」はなぜTFTに対して起きないのか

- 状態:**設計確定・単体テスト済み・本番データでの実行待ち(Mac)**
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

**(Macでの本番実行待ち。実行後、このセクションに以下を追記する:)**

- `markov_selfplay_widest_path.csv`:SELF/POP/WSLS相手の底→和・和→底のwidest pathの
  世代ごとの変化(TFT相手と比較し、(a)回復力の欠如か(b)仲間限定の回復力かを判定する)。
- `markov_selfplay_recovery_time.csv`:TFT・SELF・POP・WSLS相手の「底からの回復時間」の
  世代ごとの中央値・未回復率。もし進化した個体がSELF/POP/WSLS相手では速く回復し、TFT相手
  でだけ回復しないなら、(b)仲間限定の回復力の仮説を支持する。もしSELF/POP/WSLS相手でも
  回復しないなら、(a)回復力そのものの欠如を支持する。
- `markov_selfplay_edges.json`・`markov_selfplay_nodes.json`:ビューアでTFT・SELF・POP・
  WSLSの3-gram網を並べて見るための構造データ(今回のビューア本体の改修は別途必要;
  このファイルはデータの書き出しのみ)。
