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

---

## 8. 段階2:安念の判断(2026-10-23)と、それに基づくパイロット設計

### 8-1. 安念の判断(5点、確定)

1. **本番データ**:候補(A)結合進化(非空間)を使う。候補(B)空間版は今回扱わない。
2. **固定戦略の相手の表現**:1-4節の案(b)——固定戦略(TFT・AllC・AllD・Random・
   Gradual・WSLS)はノイズなしでtableを直接読むだけ(Kの層を通さない)。
   SELF(同じ個体)・POP(同じ集団から頻度比例で選んだ個体)は、両方とも
   Kの層を通す(focal側・相手側とも`Genome`として`update_internal_state`を通る)。
3. **類型のしきい値**:K_self・K_outの中央値を、このKモデルの本番データ全体
   (gen 1000以降をプール)で**一度だけ**計算して固定する(案1の変種——
   3節の空間版しきい値[.7770470/.5025085]は流用しない、Kモデル自身のデータで
   計算し直す)。パイロットではパイロットのデータで計算し、値と算出手順を記録する
   (本番では本番データで計算し直す——固定するのは**手順**であって**数値そのもの**
   ではない)。世代ごとの中央値で切る版(3節案2)も感度分析として併せて出す。
4. **遺伝子型の一意性判定**:K_self・K_outも含める(8-3節で具体案)。
5. **類型→人の群の対応**:K-安定型(高K_self・高K_out)→Secure、
   K-拒絶型(高K_self・低K_out)→Mix、K-とらわれ型(低K_self・高K_out)→Mix、
   K-恐れ型(低K_self・低K_out)→Fearful(K_self高=見捨てられ不安が低い、
   K_out高=親密性回避が低い、という安念の解釈——4類型はBartholomew &
   Horowitz[1991]の成人愛着4分類[安定型・拒絶型・とらわれ型・恐れ型]に
   そのまま対応する。表記は`docs/PREREGISTRATION_k_model_markov.md`1-1節:
   色の名前[白・青・緑・ピンク]は図の配色としてのみ使い、本文・コードでは
   「K-」を冠した愛着型の名前[コード識別子はK_Secure・K_Dismissing・
   K_Preoccupied・K_Fearful]で呼ぶ)。
   データ・網は**4類型のまま**作る。人と比べるときだけ、K-拒絶型+K-とらわれ型を
   合成したMixも作る。抽出人数は人の群に合わせる(Secure 56・Mix 46・
   Fearful 69)。

### 8-2. 流用方式の決定:「統合」でなく「移植」

`claude/markov-production-analysis`(先端`ab6584795340911b975a694f5b059b07b41f13f1`)の
`evopersonality_abm/markov_recovery.py`・`markov_apply.py`・`markov_vocab/*.txt`・
`docs/human_network_weights.csv`・対応するテストを、`git show <commit>:<path>`で
読み出して**このブランチに一字一句そのままコピー**した(コミット`60aaf9cd`、
diffで確認済み)。`git merge`による統合は**行わなかった**。理由:

- `git merge`は、分岐元(`ecb57e29`)以降に`claude/markov-production-analysis`が
  積み上げた**無関係な別研究(純Lindgrenのマルコフ構造進化)のコミット履歴全体**
  (本番の進化結果CSV・確証的結果JSON・ビューアのHTML・各種PNGなど、数十〜百MB
  規模の大きなデータファイルを含む)を、このKモデルのブランチに丸ごと持ち込む。
  2つの研究系統の履歴が絡み合い、どちらのブランチも見通しが悪くなる。
- 移植する2ファイル(`markov_recovery.py`・`markov_apply.py`)は**自己完結**
  しており(`probe.py`等、このブランチにまだ無いものへの依存が一切ない)、
  移植は機械的(内容を変えずコピーするだけ)。
- 欠点(両論併記):移植後に`markov-production-analysis`側でこれらのファイルに
  バグ修正が入っても、このブランチには自動で反映されない。今後そのような
  修正が入った場合は、手動で再度`git show`して差分を確認し、必要なら
  このブランチにも反映する(本メモにその都度記録する)。

### 8-3. 遺伝子型の一意性判定:連続値のKをどう扱うか(安念への提案)

1-4節で提起した論点。安念の判断(8-1節4点目)により「K_self・K_outを一意性
判定に含める」ことは決まったが、**連続値ゆえにほぼ全個体が別遺伝子型になる
場合の扱い**(個体ごとに扱うか、丸めるか)について、以下の案を提示する。

**事実確認**:`coupling.mutate_genome`は、tableの変異(`point_mutation_rate`等、
確率的——変異が起きない個体も多い)とは違い、**K_out・K_selfには毎世代・
全個体に無条件でガウス的な微小変異(`k_mutation_std`、既定0.05)をかける**
(`new_k_out = clip(genome.k_out + gauss(0, k_mutation_std))`、
`coupling.py:186-187`)。つまり、無性生殖で複製されただけの個体同士でも、
tableは複製元と完全一致しうる(変異が起きなければ)が、K_out・K_selfは
**毎世代ほぼ確実にわずかに変化する**。したがって「K_self・K_outも含めた
完全一致での一意性判定」は、安念の見立てどおり、実質的にほぼ全個体が
別々の「遺伝子型」(頻度1)になる。

- **案1(推奨):既存の`genotype_key`をそのまま拡張する——丸めない**。
  純Lindgren側の`probe.genotype_key`は`(memory_depth, table, prehistory)`の
  **完全一致**でユニークな遺伝子型を数える(変異した子は新しい遺伝子型になる、
  という最も素直な定義)。これを`(memory_depth, table, prehistory, k_out, k_self)`
  に拡張し、**K_out・K_selfも完全一致(丸めなし)で**判定する。
  - 長所:新しいパラメータ(丸め幅)を導入しない。無変異の複製(tableが
    一致し、かつ偶然K_out・K_selfも一致——通常は起こらない)は正しく
    1つの遺伝子型として数えられ、それ以外は正しく別々に数えられる、という
    既存の定義の自然な拡張に過ぎない。
  - 短所:実質的にほぼ全個体の頻度が1になる(= 後述の
    `resample_cross_seed_group`等の頻度重み付き抽出は、事実上「集団から
    一様に抽出する」のと同じになる——情報は失われない。数学的に見れば、
    頻度1の遺伝子型をプールから頻度重み付きで引くことと、元の個体リストから
    一様に引くことは同じ分布になる)。計算コストの節約(同じ遺伝子型を
    1回だけプローブして使い回す)がほぼ効かない——ただし候補(A)は
    個体数100規模(1-2節参照)なので、全個体を毎回プローブしても
    パイロット・本番とも計算量は問題にならない見込み(8-6節の見積もり参照)。
- **案2:K_out・K_selfを丸めてから一意性判定する**(例:小数第2位)。
  tableが同じでK値が近い個体をまとめて数えられるが、丸め幅という恣意的な
  パラメータが増え、丸め方次第で「同じ型」の境界が変わってしまう
  (量子化誤差で、本来は別の個体がまとめられたり、僅差の個体が割れたりする)。
  候補(A)の個体数規模では計算量を節約する必要性も薄い。
- **案3:遺伝子型の一意性判定・頻度の仕組み自体を使わない**(census層を
  飛ばし、世代の個体リストから直接・一様に抽出する)。案1と数学的には
  同じ結果になる(頻度1の遺伝子型を頻度重み付きで引くのと、個体を一様に
  引くのは同じ)が、既存の`census_genotypes`/`resample_*`の枠組み
  (頻度の列を持つプール形式)に合わせておいたほうが、6-1節で挙げた
  移植コードとの整合性が高く、「表(table)だけが一致した複製」を
  正しく1個体として数えられる(案3は複製かどうかを区別しない)。

**採用**:**案1**を採用し、8-4節以降のコード・パイロットはこれに基づいて
実装する(低リスクな選択——後から丸め[案2]に切り替える場合も、
`genotype_key`関数を1箇所差し替えるだけで済む)。安念の確認・変更の
指示があれば差し替える。

### 8-4. ノイズの条件の違い(文書化のみ、依頼どおり)

Kモデルには`coupling.run_coupled_duel`自体に**既定`noise_std=0.05`の
ガウスノイズ**が、`s`の更新式に組み込まれている(1-2節の式、`r`・`a`を
混ぜた後にガウスノイズが足される——状態そのものに乗るノイズ)。これは
純Lindgrenの本番条件(`noise_rate=0.01`、行動そのものを独立確率0.01で
反転させる「実行の誤り」——`duel.run_duel`のdocstring参照)とは**種類も
大きさも異なる**:

- Kモデルの`noise_std=0.05`:毎ラウンドの**内部状態**`s`にガウスノイズが
  足される。行動は`Bernoulli(s)`から確率的に決まるため、ノイズは行動の
  決定機構そのものに組み込まれている。
- 純Lindgrenの`noise_rate=0.01`:tableの決定(決定論的)で行動を決めた**後**、
  独立に0.01の確率でその行動をひっくり返す。決定機構とは別の層。

このKモデルのプローブでは、**Kモデルの個体どうし(SELF・POP)の対戦には
Kモデル自身の既定ノイズ(`noise_std=0.05`)をそのまま使う**(Kモデルの
通常の対戦条件と揃える——これを0.01に変更する理由がない)。**固定戦略
相手(TFT等)側はそもそも8-1節2点目の判断(b)によりKの層を通らず、
決定論的にtableを読むだけなので、相手側にはノイズが一切かからない**
——focal側(Kモデルの個体)だけが内部状態の揺らぎを持つ、非対称な設定になる。
これは純Lindgrenのプローブ(両者に対称に`noise_rate`がかかる)とは
構造的に異なる点であり、結果の解釈時に混同しないよう、パイロット・本番の
報告書には必ず明記する。

### 8-5. 注意書き

図・報告書には、動画と全く同じ文言を必ず添える(5節と同じ):

> 中央値で無理やり分けた型であり、集団は連続した一つの雲

### 8-6. パイロットの設計

- **本番データ**:`evopersonality_abm.coupling.run_coupled_evolution`を直接
  ループとして再実装し(`run_lindgren_markov_sweep.py`と同じ方式——
  `coupling.run_coupled_evolution`をブラックボックスとして呼ぶのではなく、
  世代ループを展開して中間の`genomes`リストにアクセスできるようにする)、
  既存の`scripts/run_coupling_seed_sweep.py`系の出力は一切書き換えない、
  別の新しい出力フォルダに書く。
- **記録する遺伝子型情報**:`genotype_id, frequency, memory_depth, table, prehistory,
  k_out, k_self`(既存のcensus.csv.gzの列にk_out・k_selfを追加した形)。
- **プローブする相手**:TFT・AllC・AllD・Random・Gradual・WSLS(固定、8-1節
  2点目の方式(b))+ SELF + POP(ともにKの層を通す)。
- **パイロット規模**:3〜5 seed、個体数100(候補(A)の既定)、世代はgen 1000
  以降を中心に(しきい値計算がgen 1000以降のプールを前提とするため)。
  世代ごとの個体数(100)×プローブ回数が、純Lindgrenのパイロット
  (10 seed規模)より小さいため、クラウドで計算可能と見込む——実測は
  8-7節で報告する。
- **出力**:相手×類型×世代で、64語彙の網(辺の重み・ノードの出現率)・
  widest path・底からの回復時間(戻らなかった割合・戻った個体の中央値)を、
  既存のビューア用JSONと同じ形式で書き出す。類型ごとの個体数(世代ごと)も
  別途報告する(類型によっては網を作るのに個体数が足りない可能性を確認するため)。
- **結果の解釈はしない**:パイロットは出力の形・量・計算コストを確認する
  段階であり、どの類型がどの人の群に近いかといった解釈は、安念がパイロット
  結果を見たあと、段階3(事前登録の草案)で行う。

### 8-7. パイロットの実行結果(実測、解釈なし)

実行:このクラウドセッションで完結(Mac不要)。`scripts/run_coupling_markov_pilot.py`
(4 seed・seed 100〜103、個体数100、1200世代、gen 1000〜1200を50世代おきに5時点
プローブ、ワーカー4)。

**所要時間**:4 seed合計で約36分(進化:seedあたり約1.2秒/世代実測どおりの
ペース、4ワーカー並列)。`scripts/compute_k_model_type_thresholds.py`は数秒。
`scripts/export_k_model_markov_viewer.py`(5時点×8相手×5群[4類型+Mix]×
200回抽出)は約2分9秒。8-6節の見積もりの範囲内に収まった。

**しきい値**(`type_thresholds.json`、gen≥1000をプール、n=2000=4 seed×5時点×
100個体):threshold_kself = 0.7308933、threshold_kout = 0.4319747
(このパイロット自身のデータから算出した値——3節で述べたとおり、空間版の
しきい値[.7770470/.5025085]とは別物で、本番ではこの値も本番データで
算出し直す)。

**類型ごとの個体数(世代ごと、`k_model_type_counts.csv`)**:4類型とも、
5時点のどこでも0に近づくことはなかった(最小はK-安定型、gen1000の43/400個体;
他はおおむね70〜180/400の範囲で変動)。8-6節で確認したかった「類型に
よっては個体が少なすぎて網が作れない」という懸念は、**この4 seed・
population=100規模のパイロットでは起きなかった**(本番でもっと多い
seed数・世代数で同じ傾向が続くかは別途確認が必要)。

**出力の形・量**:
- `k_model_widest_path.csv`:200行(8相手×5群[K-安定型・K-拒絶型・K-とらわれ型・
  K-恐れ型・Mix]×5時点、欠損なし=期待される全組み合わせが揃っている)。
- `k_model_recovery_time.csv`:160行(8相手×4類型[Mixは除く——Mixは網の
  比較専用の合成群であり、回復時間はK-拒絶型・K-とらわれ型それぞれで既に
  数えているため二重に数えない]×5時点)。
- `k_model_edges.json`・`k_model_nodes.json`:8相手×5群×5時点、すべてに
  動画と同じ注記文言(`CAVEAT_TEXT`)を含む。ファイルサイズはedges 696KB・
  nodes 160KB。

**サンプリング方式の注記の再確認**:8-6節で述べたとおり、4 seedでは
`markov_apply.resample_cross_seed_group`(系統間抽出、distinct seedが
group_size以上必要)が使えないため、全seedの該当類型の個体をまとめて
1つのプールにし(`merge_pools_across_seeds`)、そこから頻度重み付き・
復元抽出で引く方式(`resample_within_seed_group_with_identity`)を使った。
本番(seed数が人の群の人数[46〜69]以上に増えたら)は系統間抽出に
切り替えるべきかを検討する。

**結果の解釈(どの類型がどの人の群に近いかなど)はまだ行っていない**——
依頼どおり、安念がこのセクションを見たあと、段階3(事前登録の草案)で
行う。

## 9. 段階3:事前登録の確定

`docs/PREREGISTRATION_k_model_markov.md`は、以下の確定コミットで
草案から確定に切り替わった:

- **確定コミット:`<PENDING_COMMIT_HASH>`**
- 確定時に加えた主な変更:類型名の愛着理論名への統一(本メモ8-1節・8-7節の
  改名を含む)、H-A(対応の特異性)を連結ベクトルでの単一相関に改定、
  H-B(回復)への仕組み上の懸念の明記、H-C(記憶の深さ)の層の切れ目・
  最低人数(20、確定)の追加、置換検定の方式を(b)案(置換1000回×
  1置換あたりdraw 50回、H-A)/置換1000回・draw制約なし(H-B・H-C)に確定、
  および置換検定内で本物のラベルの統計量を置換と同じdraw数で計算し直す
  手順の追加(いずれも事前登録文書自身の5節・4節に詳細がある)。
- 本番実行の手順は`docs/K_MODEL_MARKOV_MAC_RUNBOOK.md`にまとめた。
