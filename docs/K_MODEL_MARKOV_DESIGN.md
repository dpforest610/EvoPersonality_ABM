# 設計メモ:Kモデル個体のマルコフ構造プローブ(段階1)

- 状態:**設計メモのみ、未実行**(このメモに基づくコード実行・パイロットは安念の確認後、段階2で行う)
- 対象ブランチ:`claude/k-model-markov-probe`
- **分岐元は`main`ではない。** `main`(`635ab597`、2026-08-28)には二成分Kモデル
  (`evopersonality_abm/coupling.py`等)がまだ無い。このブランチは
  `claude/inspiring-allen-ja4ele`の先端コミット**`ecb57e29`**
  (「Add N=1000 production raw individual logs (5 conditions x 30 seeds; pilot excluded)」、
  2026-10-14)から分岐した。`main`は746コミット遅れている。
- Lindgren側(マルコフ構造・3-gram網・widest path・ビューア)のコードは
  `claude/markov-production-analysis`(先端コミット**`ab6584795340911b975a694f5b059b07b41f13f1`**、
  2026-10-22)にある。**このブランチには段階1では統合しない** — 以下は
  `git show origin/claude/markov-production-analysis:<path>`で読んだ内容の参照のみ。
  統合するか、Kモデル用に書き直すかは段階2で決める。
- 人のデータ(質問紙・個人の3-gram系列)はこのリポジトリに入れない。
  人側は`docs/human_network_weights.csv`(既にリポジトリにある、辺の重みのみ)だけを使う。

---

## 1. Kモデルの個体が固定の相手と対戦するとき、何が状態として持ち越されるか

`evopersonality_abm/coupling.py`を読んだ内容(このブランチに実在するファイル、
行番号は2026-10-22時点)。

### 1-1. 個体の遺伝子型(進化で決まり、プローブ全体で不変)

`Genome`(`coupling.py:52-58`)は3つだけ:
- `table`: `LindgrenTable`(純Lindgrenモデルと同じ、戦略の決定表)
- `k_out`: 自動channel(相手の直近の観測行動に引っ張られる強さ)
- `k_self`: 熟慮channel(自分のtableの推奨にどれだけ従うか)

この3つが「個体の識別子」であり、進化後は固定。純Lindgrenモデルの
`LindgrenTable.table`/`prehistory`に相当する役割。

### 1-2. 1回の対戦(duel)の中だけで持ち越される状態

`run_coupled_duel`(`coupling.py:79-142`)は対戦のたびに**ゼロから**開始する:
- 内部状態 `s_a`/`s_b` — `INITIAL_STATE = 0.5`(`coupling.py:48`)からスタート
- 相手の直近の観測行動 `last_action_a`/`last_action_b` — `None`からスタート
  (最初のラウンドは`NEUTRAL_PRIOR = 0.5`、`coupling.py:49`)
- 自分の実際の行動履歴 `history_self_*`/`history_opp_*` — 空からスタート
  (`table.decide()`の入力に使う。これは**自分が実際に行った行動**の履歴であり、
  tableの「推奨」そのものの履歴ではない — `s`に応じたベルヌーイ試行で実際の
  行動が決まるため、tableの推奨と食い違うことがある)

毎ラウンドの更新(`coupling.py:71-76`、`update_internal_state`):

    s(t) = clip( s(t-1) + k_self*(r - s(t-1)) + k_out*(a - s(t-1)) + noise, 0, 1 )
    action(t) ~ Bernoulli(s(t))

`r`は自分のtableの推奨(自分の実際の履歴から`decide()`で計算)、`a`は相手の
直近の実際の観測行動、`noise`はガウスノイズ(既定`noise_std=0.05`)。

### 1-3. 対戦をまたいでは一切持ち越されない

同じ個体が複数回プローブ対戦する場合(反復30回、相手を変える場合)、
`s`・直近観測行動・履歴はすべて対戦ごとにリセットされる。純Lindgrenモデルの
プローブ(`evopersonality_abm/probe.py`)と同じ前提(遺伝子型だけが対戦間で
不変)。

### 1-4. 未決定の設計上の論点:固定戦略相手(TFT等)をどう表現するか

`run_coupled_duel`は**両者ともGenome**であることを前提にしている
(`genome_a.table.decide(...)`と`genome_b.table.decide(...)`を対称に呼ぶ)。
固定戦略(TFT・AllC・AllD・WSLS、すべて`LindgrenTable`で表現済み — 
`evopersonality_abm/lindgren.py`の`tit_for_tat_table()`等)をプローブの相手に
使う場合、2通りの表現が考えられ、**まだ決めていない**:

- **(a) 固定相手もGenome化する**:`Genome(table=tit_for_tat_table(), k_out=0, k_self=1)`
  のように、「完全に熟慮channelのみ(k_self=1)、自動channelなし(k_out=0)」と
  みなす。この場合、固定相手側にも`update_internal_state`のガウスノイズが
  かかるため、**純LindgrenのTFT(決定論的)とは厳密には一致しない**
  (noise_stdを0にすれば一致するが、それはKモデルの標準設定から外れる)。
- **(b) 固定相手はLindgrenTable.decide()を直接読む**:Kモデルの個体
  (focal側)だけがK層を通り、固定相手はK層を経由しない。既存のLindgren
  プローブでの固定戦略の定義(常に決定論的)とより整合するが、
  focal側と相手側で扱いが非対称になる新しい関数が要る
  (`run_coupled_duel`を改造、または専用の新関数)。

どちらで行くか、あるいは両方試して比較するかは、安念の判断を仰ぐ。

---

## 2. 本番データの候補:どちらを使うか

### 候補(A): 結合進化(非空間)

`evopersonality_abm/coupling.py`の`run_coupled_evolution`、
`scripts/run_coupling_seed_sweep.py`(複数seed並列実行)。空間構造なし、
集団内で総当たり(`round_robin_payoffs_coupled`、`coupling.py:145-161`)。

既に`scripts/output/coupling_seed_sweep_5000gen/`に**50 seed分**の
`seed_X_timeseries.csv`があるが、**これは世代ごとの集計(K_out/K_selfの平均・
分散、平均深さ、平均payoff)のみ**。個体の`LindgrenTable`の実体(table・
prehistory のビット列)は保存されていない。

### 候補(B): 空間版(N=1000)

`evopersonality_abm/scent_spatial.py`、`scripts/run_scent_spatial_sweep_n1000.py`。
13×13を超える格子上でのローカルな近傍対戦+移動(none/random/scentの3条件)。

このクラウド環境に既に`scripts/output/scent_spatial_sweep_n1000/{none,random,scent}/`
以下、**30 seed×3条件**の本番の生ログがある(直近の「Add N=1000 production
raw individual logs」コミット)。`seed_X_individual_log.csv.gz`の列を確認した:

    generation,agent_index,k_out,k_self,memory_depth,payoff,
    opponent_mean_k_self,opponent_mean_k_out,pos_x,pos_y

**`memory_depth`は数値のみで、`LindgrenTable`の表そのもの(ビット列)は
含まれていない。** 空間的な近傍構造・移動の条件(M0/M1/M2)という複雑さが
あり、プローブに座標や匂いの場は不要。

### 共通の結論(重要): どちらも今あるデータだけではプローブできない

両候補とも、完走後の**チェックポイントは削除される設計**
(`run_scent_spatial_sweep_n1000.py`: `checkpoint_path_for(out_root, seed).unlink(missing_ok=True)`、
seed完走時に削除。`coupling`系の実行も同様の設計)。チェックポイントには
完全な`Genome`のリスト(`ckpt["genomes"]`)が入っているが、**完走後は
消えている**。つまり:

**遺伝子型(table・prehistory)を世代ごとに保存する、新しい記録付きの実行
(または既存の記録コードへのtable保存の追加)が、候補(A)(B)どちらを選んでも
必要になる。**

### 長所・短所

| | (A) 結合進化(非空間) | (B) 空間版(N=1000) |
|---|---|---|
| 計算コスト | 低い(総当たり、100体規模) | 高い(N=1000、近傍探索・移動あり) |
| 既存データの蓄積 | 50 seed(集計のみ) | 30 seed×3条件(個体ログ、tableなし) |
| 構造の単純さ | 単純(空間構造なし) | 複雑(近傍・移動条件という余分な次元) |
| Kモデルの「型」が意味を持つ文脈 | 弱い(全員と対戦するので局所的な同類性なし) | 強い(近傍の同類性・移動条件で型の構成が変わりうる、動画の題材そのもの) |
| プローブ対象の選定 | 全個体から一様 | 条件(M0/M1/M2)ごとに分けて見る必要がある分、設計がもう1段複雑 |

どちらを選ぶかは安念の判断。(B)は動画・既存の型分けしきい値の実績
(下記3節)と直接つながる一方、(A)は実装・検証コストが低い。

---

## 3. 類型の切り方

### 案1: 固定しきい値

依頼文にあった`K_self ≥ 0.7835, K_out ≥ 0.4566`は、
`docs/SCENT_MODEL_N1000_PHASE_ANIMATION.md`(269-271行)に明記されている
**「試作で使ったseed 0だけの値」であり、既に置き換え済み**。現在、本番の
Heatbugs動画・学会用の図で実際に使われている固定しきい値は:

    K_self 中央値(threshold_kself) = 0.7770470
    K_out  中央値(threshold_kout)  = 0.5025085

(`scripts/compute_scent_n1000_type_thresholds.py`で、全30 seed×3条件[M0/M1/M2]・
gen≥2000をプールして一度だけ計算・固定したもの。
`scripts/output/scent_spatial_sweep_n1000/analysis/type_thresholds.json`に保存済み。)

このKモデル・マルコフプローブでも、この既存の固定値をそのまま使うか、
Kモデル・マルコフプローブ用のデータ(2節で決める候補)に対して改めて
中央値を計算し直すかは検討が必要(候補(A)[結合進化]を選ぶ場合は、
この固定値の算出元[候補(B)の空間データ]とは別の集団なので、流用が
適切かどうかの判断がいる)。

### 案2: 世代ごとの中央値

固定しきい値を使わず、世代ごとにその世代の集団のK_self・K_out中央値で
4類型に分け直す(動的な分割)。集団全体のK分布が世代とともに動く場合、
相対的な位置関係を保つ一方、絶対値での比較(例えば「浅い世代のタイプAと
深い世代のタイプAは同じ基準か」)が難しくなる。

両案とも決定しない。安念の判断を仰ぐ。

---

## 4. 4類型→人の3群への対応表(案のみ)

既存の4類型(`docs/SCENT_MODEL_N1000_PHASE_ANIMATION.md`の定義):
白=高K_self/高K_out、青=高K_self/低K_out、緑=低K_self/高K_out、
ピンク=低K_self/低K_out。

人の愛着群は3群(Secure・Mix・Fearful)なので、4→3の対応は1対1にならない。
**安念が決めるので、ここでは案を並べるだけで決めない**:

- **案a**: 4象限のうち3つを人の3群にそのまま対応させ、残り1つは
  「対応する人の群なし」として別扱いにする(どの象限を除外するかも未定)。
- **案b**: K_self軸だけで3分位(高・中・低)に分け、K_outは今回の対応表では
  使わない(愛着理論の回避[avoidance]軸をK_out、不安[anxiety]軸をK_selfに
  見立てるなら、不安軸寄りの3分割に相当するかもしれない、という仮説レベルの
  対応)。
- **案c**: K_self・K_outの2軸から何らかの合成得点(例:差・比・主成分の
  第1軸など)を作り、その1次元で3分位に分ける。

---

## 5. 注意書き(動画と同じ文言)

Heatbugs動画・関連図と**全く同じ文言**を、このプローブの図・報告にも
必ず添える(`docs/SCENT_MODEL_N1000_PHASE_ANIMATION.md`306行目から一字一句):

> 中央値で無理やり分けた型であり、集団は連続した一つの雲

---

## 6. 流用できるコード・書き直すコードの一覧

### 6-1. 流用できる(Lindgren側の「ものさし」— `claude/markov-production-analysis`、コミット`ab6584795340911b975a694f5b059b07b41f13f1`、`git show`で参照のみ、このブランチには未統合)

- **`evopersonality_abm/markov_recovery.py`**: `series_from_moves`・
  `build_edges`・`widest_path_threshold`は、ユーザーが添付した
  `markov3_widest_path_reference.py`(人の研究のコード、Annen JOEP論文)から
  **一字一句変更せず移植**されたもの(ファイル冒頭のdocstringに明記、
  2026-10-15に人のデータN=171・Table 15/Table 8と3桁一致で検証済み)。
  `self_loop_edges`(自己ループの重み、R版`GRA_MAR10feb04Fmarkov_network_with_loops.R`の
  自己推移検出ロジックの移植)、`NAMED_CIRCUITS`(Table 7の固着回路)、
  `load_vocab`(64語彙の読み込み:`evopersonality_abm/markov_vocab/*.txt`)も
  ここにある。**3-gram網・widest path・自己ループの定義そのものは、
  相手がKモデルの個体でも人でも固定戦略でも変わらないはずなので、
  このファイルは変更せずそのまま使えると見込む。**
- **`evopersonality_abm/markov_apply.py`**: `resample_virtual_group`・
  `resample_cross_seed_group`・`bootstrap_aggregate_with_sampler`・
  `network_metrics`など、「仮想参加者を抽出して網を作り指標を集計する」
  一式。系統間抽出(56人)の仕組みもここ。**抽出元のプールの作り方
  (今は`{seed: (frequencies, series_by_genotype, identities)}`)さえKモデルの
  本番データに合わせて用意できれば、集計ロジック自体は流用できる見込み。**
- **`scripts/export_markov_edge_weights_for_viewer.py`**: 64語彙の辺の重み
  (系統間抽出200回の平均Jaccard)と、`--nodes-only`オプションによる
  ノードの出現率(同じ200回の抽出、各3-gramを含む仮想参加者の割合)を、
  ビューア用JSONとして書き出す一式。**出力スキーマ(`{相手: {群: {世代: ...}}}`)を
  そのまま踏襲し、相手名に"SELF"/"POP"/"WSLS"(前の依頼)や、Kモデルの
  4類型名を追加する形で流用できる見込み。**
- **`scripts/export_depth_groups_for_r.py`**・**`scripts/plot_markov_depth_networks.py`**:
  個体を何らかの基準(記憶の深さ)で複数の群に分け、群ごとに3-gram網を作る、
  という枠組み自体は、Kモデルの4類型分けと構造的に同じ(「連続値を閾値で
  区切って複数の仮想集団にする」という操作)。**区切り方の関数を差し替えれば
  流用できる見込み。**

### 6-2. 書き直す必要があるもの

- **プローブ実行の中核**: `evopersonality_abm/probe.py`の`OPPONENTS`
  (固定戦略の定義)自体は変更不要だが、「Kモデルの個体を、固定戦略
  (または他のKモデル個体)と対戦させてプローブする」仕組みは新規。
  1節で未決定の、固定相手の表現方法(Genome化するか、tableを直接読むか)を
  先に決める必要がある。
- **本番の記録**: `coupling.py`/`scent_spatial.py`系の実行は、現状
  世代ごとの個体の`table`・`prehistory`を保存していない(2節参照)。
  プローブ可能な記録を残す新しい実行(または既存コードへの記録追加)が要る。
- **census相当の一意性判定**: Lindgren側は`(memory_depth, table, prehistory)`の
  組でユニークな遺伝子型を数える(`evopersonality_abm/probe.py`の
  `genotype_key`)。Kモデルでは`table`が同じでも`k_out`・`k_self`が違えば
  別の表現型になるため、**一意性の定義に`k_out`・`k_self`を含めるかどうか
  (含めないと、Kの値が違う個体が同一遺伝子型として扱われ、網の構築時に
  Kの違いが消えてしまう)を決める必要がある** — 段階2で検討。

---

## 7. 段階2・3への申し送り

- 段階2は、安念がこの設計メモ(特に2節の本番データ候補・3節のしきい値・
  4節の対応表)を確認した後に着手する。3〜5 seedのパイロットで、
  4類型それぞれの網(TFT・AllC・AllD・Random・Gradual)をビューア用JSON
  (相手×類型×世代、辺とノードの出現率)で書き出す。人の群の人数
  (56・46・69)に合わせた抽出。
- 段階3は、パイロットの後に事前登録の草案を作る。仮説は人の群との対応
  (どの類型の網がどの群に近いか)。
