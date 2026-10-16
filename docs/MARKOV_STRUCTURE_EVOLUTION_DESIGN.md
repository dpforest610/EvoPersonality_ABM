# 新研究ライン: Lindgren モデルにおける行動マルコフ構造の進化 — 設計メモ

この文書は設計メモであり、実装はまだ行っていない。ここに書いた提案についてご確認・ご指示を
いただいてから実装に進む(事前登録の本文もその後)。

---

## 0. 問いの確認

記憶の深さが進化していく Lindgren の集団の中で、相互裏切り(DD)が3回続いた状態
(「底底底」)からの回復過程が、進化とともにどう変化するかを、固定した標準の相手との
対戦(プローブ)を通して追う。人の実験(JOEP 原稿)と同じ3ステップマルコフ網・
widest path 法で、回復力を定量化する。

---

## 1. 進化モデル: 使う実装とその理由

### 1.1 既存実装の棚卸し

リポジトリには3つの対戦・進化の層がある。

| 層 | ファイル | K | ノイズ | 用途 |
|---|---|---|---|---|
| 素の Lindgren | `agent.py`/`duel.py`/`strategies.py`/`population.py` | なし | **なし** | 第一章(ラベルなしだが`docs/LOCAL_RUN.md:265`「第一章k=99比較」が示唆)。`population.py`冒頭のdocstringが明記:「No spatial structure, no odor gradient, no K_out/K_self coupling yet...この module is the population-level foundation」 |
| K結合 | `coupling.py` | あり(`Genome.k_out`/`k_self`) | あり(`noise_std`、連続状態への加法ガウスノイズ→ベルヌーイ採択) | 第二章相当(HBES2026骨子) |
| 空間+匂い | `scent_spatial.py` | あり | あり(coupling.pyと同じ機構) | 第三章 |

今回は「K のない純粋な Lindgren モデル」なので、**`evopersonality_abm/population.py`の
`run_evolution()`を使う**(`LindgrenTable`による記憶ゲノム、`lindgren.duplicate_memory`/
`split_memory`/`point_mutate`、`population.select_and_reproduce`による適応度比例選択)。
これが実際に「K を一切import しない」唯一の進化ループであり、使う実装はこれと明記する。

### 1.2 ギャップ: ノイズが無い

`population.py`の対戦(`round_robin_payoffs`→`run_duel`、`duel.py`)は完全決定論的で、
手の実行にノイズが入る経路が無い(確認済み: `run_duel`は`agent.decide()`の結果をそのまま
使うだけ)。一方、依頼には「ノイズあり」と明記されている。Lindgren (1991) 自身のモデルでも、
IPDの文献一般でも、この種のノイズは「実行誤り(implementation error / trembling hand)」
として、戦略が指示した手と逆の手が一定確率で実際に出る、という形で入れるのが標準。

**提案**: `duel.py`の`run_duel`に、後方互換なオプション引数を追加する
(`noise_rate: float = 0.0`, `rng: Optional[random.Random] = None` —
省略時は現状と完全に同じ決定論的動作、既存のテスト・呼び出し元は無変更):

```python
def run_duel(agent_a, agent_b, rounds=10, noise_rate=0.0, rng=None):
    for _ in range(rounds):
        action_a, action_b = agent_a.decide(), agent_b.decide()
        if noise_rate and rng.random() < noise_rate:
            action_a = DEFECT if action_a == COOPERATE else COOPERATE
        if noise_rate and rng.random() < noise_rate:
            action_b = DEFECT if action_b == COOPERATE else COOPERATE
        payoff_a, payoff_b = payoff(action_a, action_b)
        agent_a.record(action_a, action_b, payoff_a)
        agent_b.record(action_b, action_a, payoff_b)
    return agent_a, agent_b
```

ノイズは「実際に記録される手」だけを変える(ゲノムの表が指示した手は変えない。次の
ラウンドの参照履歴には、実際に出た手が入る — これは`coupling.py`のK結合ノイズと
同じ考え方)。

**確認したい点(1)**: ノイズ率の値。リポジトリにこの種の「実行誤り率」の先例は無い
(K結合の`noise_std=0.05`は連続状態への加法ノイズで、スケールが違うので直接は使えない)。
Lindgren (1991) や関連文献でよく使われる1%前後を仮提案する
(`point_mutation_rate=0.05`と同程度〜やや小さい)。ご指示があれば変更する。

### 1.3 パラメータ提案

| パラメータ | 提案値 | 理由 |
|---|---|---|
| population_size | 100 | 後述のプローブで「集団中の各遺伝子型を頻度で重みづける」ために、
  個体数が少なすぎると頻度の解像度が粗くなる。`run_seed_sweep.py`・coupling章の
  いずれも100を使っており、本リポジトリの事実上の標準。 |
| rounds_per_duel | 50 | `population.run_evolution`の関数既定値は150だが、既存の本番実行
  (`scripts/output/seed_sweep_5000gen/`, coupling章)はいずれも50を使っている。
  世代あたりの対戦コストを抑えつつ、既存の実行コストの見積もり(1.2節参照)と
  直接比較できる値として50を提案。 |
| initial_memory_depth | 1 | 関数既定値のまま。 |
| point_mutation_rate | 0.05 | 既定値のまま(coupling章と同じ)。 |
| duplication_rate | 0.02 | 既定値のまま。 |
| split_rate | 0.02 | 関数既定は0.0(重複のみ、深さは増える一方)だが、coupling章・scent章は
  いずれも0.02を使っており、「記憶の深さが進化していく」(増減とも)過程を見たいので
  split_rateを入れることを提案。 |
| max_memory_depth | 6 | `run_population_evolution.py`の既定と同じ。表サイズは4^depth
  なので、6で4096エントリ(妥当)。8だと65536で重くなる。 |
| min_memory_depth | 0 | 既定のまま。 |
| noise_rate(新規) | 0.01(仮) | 1.2節参照、確認事項(1)。 |

### 1.4 世代数・seed数(断続平衡が見える規模)

既存の記録(`scripts/output/seed_sweep_5000gen/summary.csv`、population=100,
rounds_per_duel=50、ノイズなし)では、5シードのうち2シードだけが5000世代以内に
「裏切り→協調」のレジーム転換を起こした(世代17と世代2582)。残り3シードは
5000世代でも転換しなかった。**この数字はノイズが無い実行のものなので、ノイズを
入れたこの研究の実際の動態とは異なりうる**(ノイズは単純な全協調・全裏切りの均衡を
不安定化させ、転換をむしろ起こしやすくする可能性がある)。

断続平衡(ここでは「底底底が繰り返し現れては回復する」ような、1回限りでない往復)を
複数シードで複数回観測するには、上記の実測ですら数千世代かかる(あるいは起きない)
ことを踏まえ、次の2段構えを提案する。

**パイロット(較正)**: population=100, rounds_per_duel=50, ノイズ率を1〜2通り
(0と提案値)、**10 seed × 2000世代**。目的は「ノイズありだと転換がどのくら
いの頻度・時間スケールで起きるか」を見ること。1.2節の実測で計算量を見積もる。

**本番**: パイロットの結果を見て決めるが、既存の記録と本リポジトリの先例
(coupling章・scent章はいずれも最終的に50 seed × 5000世代)を踏まえた初期提案は
**50 seed × 5000世代**。パイロットで「2000世代でも転換がほとんど起きない」と
分かれば世代数を増やす、「頻繁に起きる」なら5000のままで十分、という判断をする。

### 1.5 計算量の実測

このクラウド環境で実測(`population.run_evolution(population_size=100,
rounds_per_duel=50, generations=10, seed=0)`、ノイズなし版):

    10世代で12.4秒 -> 1.24秒/世代

5000世代では1シードあたり約103分(1.7時間)。50 seedでは**合計約86時間**
(シード間で並列化すれば、例えば8並列で約11時間、16並列で約5.5時間 —
scent章と同じ`ProcessPoolExecutor`による並列化を想定)。ノイズ処理を足しても
対戦1回あたりのオーバーヘッドは小さい(if文2回追加)ので、この見積もりから
大きくは変わらない見込み。

パイロット(10 seed × 2000世代)は、1シードあたり約41分、10シード合計で
約7時間(並列化すれば1〜2時間程度)。

---

## 2. プローブ(仮想参加者対戦)の設計

### 2.1 いつ測るか

**提案: 50世代おき**。本番5000世代なら1シードあたり101時点。理由:
- 100世代おき(scent章のLOG_INTERVAL)だと、世代17のような早い転換を
  取りこぼす可能性がある。
- 1世代おきは記録量・計算量ともに過大(2.4節参照)。
- 50は、世代17のような早い転換の近傍もある程度の解像度で捉えつつ、
  記録ファイル数を現実的に保てる折衷値として提案。

**確認したい点(2)**: この間隔でよいか。パイロットの結果(転換がどの時間
スケールで起きるか)を見てから、本番の間隔を調整してもよい。

### 2.2 プローブ対象: 遺伝子型ごと(個体ごとではない)

`LindgrenTable`は`memory_depth`・`table`(タプル)・`prehistory`(タプル)の
組で一意に決まり、ハッシュ可能(タプルなので)。各世代、集団100個体を
**(memory_depth, table, prehistory)の組で重複排除**し、各ユニーク遺伝子型の
出現数を頻度とする。初期世代は各個体がほぼ別の表を持つが(100個体、深さ1でも
16通りのtableがあるため最初はばらける)、選択が効いてくると少数の遺伝子型に
収束していくので、世代が進むほどユニーク数は減り、プローブのコストも下がる
傾向になる(2.4節の見積もりは最悪ケースとして全個体別型を仮定)。

### 2.3 相手セット

**主セット(人の実験と同じ5戦略)**:

| 名前 | 実装 |
|---|---|
| TFT | 既存`lindgren.tit_for_tat_table()`(memory_depth=1) |
| AllC | 既存`lindgren.always_cooperate_table()` |
| AllD | 既存`lindgren.always_defect_table()` |
| Random(協力確率0.5) | 新規。`LindgrenTable`では表現できない
  (手番ごとに独立なベルヌーイ試行で、履歴に依らないため)。`strategies.py`の
  既存3関数と同じ形式(`(own_history, opp_history) -> action`)で
  `random_strategy_factory(rng) -> callable`を追加提案。 |
| Gradual(なだめなし) | 新規。下記2.3.1。n=1で TFT と同じになることを
  テストで保証する(依頼どおり)。 |

**拡張セット(探索的)**:

| 名前 | 実装 |
|---|---|
| Gradual(なだめあり、原版) | 新規。Beaufils, Delahaye & Mathieu (1996):
  相手のn回目の裏切りにn回連続の裏切りで応え、そのあと2回の協調(なだめ)を
  挟んでから通常運転に戻る。 |
| WSLS | 既存`lindgren.pavlov_table()`(Win-Stay-Lose-Shift、memory_depth=1) |

#### 2.3.1 Gradual の実装設計

`Agent.strategy`は`(own_history, opp_history) -> action`という純粋関数の形を
要求するが(`agent.py:10`)、Gradualは「あと何回報復するか」「あとは何回
なだめるか」という**ラウンドをまたぐ内部状態**を持つ。この形式でも、
クロージャで状態を保持すれば実装できる(`strategies.py`の既存3関数と違って
内部状態を持つ点が新しいが、`Agent.strategy`の型シグネチャ自体は変えない)。

```python
def make_gradual_strategy(appeasement: bool) -> Callable[[list, list], str]:
    """呼び出すたびに新しい対戦用のクロージャを返す(状態は対戦ごとにリセット)。"""
    state = {"punish_remaining": 0, "calm_remaining": 0, "defections_seen": 0}

    def strategy(own_history, opp_history):
        if state["punish_remaining"] > 0:
            state["punish_remaining"] -= 1
            if state["punish_remaining"] == 0 and appeasement:
                state["calm_remaining"] = 2
            return DEFECT
        if state["calm_remaining"] > 0:
            state["calm_remaining"] -= 1
            return COOPERATE
        if opp_history and opp_history[-1] == DEFECT and not own_history:
            pass  # (このブロックは擬似コード、実際は「新しい裏切りを検知したか」の判定)
        if opp_history and opp_history[-1] == DEFECT:
            state["defections_seen"] += 1
            state["punish_remaining"] = state["defections_seen"] - 1  # 今回の手ですでに1回分を消費
            return DEFECT
        return COOPERATE

    return strategy
```

(↑設計の方向性を示す擬似コードで、実装時に「前回すでに報復中だったか」との
区別などを詰める。n=1での「相手の1回目の裏切りに1回だけ報復しすぐ戻る」が
TFTと同じ振る舞いになることをテストで確認する。)

**確認したい点(3)**: 上記の報復開始タイミング(「裏切りを検知した今回の手から
報復が始まる」)でよいか。文献によっては「検知した次のラウンドから」の版もある。

### 2.4 対戦条件

- **ラウンド数**: 21回(人の実験と同じ)。主分析は1〜20回目を使う。
  感度分析として100回版も記録する(同じ対戦をもう一度、100ラウンドで
  やり直すのではなく、**1回の対戦を100ラウンド走らせ、主分析では先頭20回だけを
  切り出して使う**方式を提案 — 21回版と100回版で手の系列が1〜20回目で
  食い違わないようにするため。21回だけの対戦をしてから別途100回版を
  走らせると、ノイズ・Randomの乱数消費がずれて2つの系列が無関係になるため、
  それぞれ独立した対戦として両方記録する)。

  **確認したい点(4)**: 上記の理由で「21回版」と「100回版」は別々の独立対戦
  (異なる乱数)として両方保存する案でよいか。それとも、1回の対戦を100ラウンド
  走らせて、主分析用に先頭20回を切り出す一本化の方が良いか(後者の方が
  対戦回数は半分で済む)。計算量的には一本化の方が有利なので、特段の理由が
  なければ一本化を推奨する。

- **ノイズ**: 進化の実行と同じノイズ率(1.2節)を主、ノイズなし(0)を感度分析。
- **繰り返し回数**: Random相手・ノイズありの両方で対戦が確率的なので、
  遺伝子型×相手の組ごとに複数回対戦し、系列のばらつきを捉える。
  **30回を提案**(人の実験のSecure群n=56等と近い桁で、後段の再抽出
  シミュレーション(3.3節)が十分な母数を持てる程度)。
- **架空の過去履歴(prehistory)**: 記憶の長い遺伝子型が最初の数手で参照する
  「対戦開始前の履歴」は、**進化で実際に使われているゲノム自身の`prehistory`
  ビットをそのまま使う**(新しく決め直さない)。理由: `LindgrenTable`は
  `duplicate_memory`で深さが伸びるたびに新しいprehistoryビットを0で埋めて
  いくが(ただし中立的に、実際の判断には影響しない設計)、各ゲノムが
  進化の過程でどんなprehistoryを持つに至ったかは、そのゲノム自身の由来の
  一部であり、プローブの時だけ別の仮定に置き換えると、進化の実行中の対戦
  (集団内総当たり)とプローブでの対戦とで同じゲノムが違う初期挙動をする
  ことになってしまう。既存実装(`lindgren.tit_for_tat_table()`等の固定
  戦略も、それぞれの戦略が「妥当な」prehistoryを個別に埋め込んでいる
  ([1,1]=相手は協調していたと仮定、AllDのみ[0,0])のと同じ原則で、
  「そのゲノム自身の履歴をそのまま使う」が一貫している。

### 2.5 プローブの計算量見積もり

ユニーク遺伝子型数を最悪50(初期世代、個体の大半が別型)、相手7種
(主5+拡張2)、繰り返し30回、1対戦100ラウンド(2.4節一本化案の場合)と
すると、1プローブ時点あたり 50×7×30×100 = 1,050,000 ラウンド。
本番(5000世代、50世代おき)なら1シードあたり101時点 ×
1,050,000 ≈ 1.06億ラウンド。1.2節の実測(1.24秒/世代、247,500ラウンド/世代、
約5マイクロ秒/ラウンド)を単純に当てはめると、1シードあたり約530秒
(約9分)。50シードで約7.4時間 — 進化本体(50シードで約86時間、並列化後
数時間〜半日)に比べ無視できない追加だが、支配的ではない規模。

実際には世代が進むとユニーク遺伝子型数が減る(選択で収束するため)ので、
これは上限寄りの見積もり。

---

## 3. 人の研究と同じマルコフ分析をこの段階に組み込む

### 3.1 移植: `evopersonality_abm`モジュール化

添付の`markov3_widest_path_reference.py`(人のデータN=171で、論文の
12値を小数第3位まで完全再現済み、2026-10-15検証)を、**定義を一切変えずに**
`evopersonality_abm/markov_recovery.py`として取り込む提案。

- `series_from_moves(focal, opp, n_rounds=20)`: 20手→18個の3-gram文字列
  (変更なし、そのまま移植)。
- `build_edges(series_list, vocab)`: 参加者ごとの3-gram**集合**
  (系列中で隣接したかどうかではなく、同じ参加者の中で両方出現したかどうか)、
  重みは参加者単位のJaccard係数。**変更なし、そのまま移植**。
- `widest_path_threshold(edges, start="底底底", goal="和和和")`:
  Jaccard値を降順にしきい値として試し、そのしきい値以上の辺だけでSTARTから
  GOALに到達可能になる最初のしきい値。START/GOALが無ければ`None`。
  **変更なし、そのまま移植**(転落の分析では`start="和和和", goal="底底底"`
  を渡せる — 関数シグネチャはそのまま使える)。

予約語ボキャブラリ(TFT 16語・GRA/RAN 64語・AllC/AllD 8語、いただいた
4ファイル — GRA/RANは確認したところ4記号3文字の全組み合わせ64通りと
完全一致、つまり「制限なし」、AllCは{和,搾}の8通り(相手が常に協調する
ので2文字目は常にC)、AllDは{惨,底}の8通り(相手が常に裏切るので
2文字目は常にD)で、定義と整合)を、`evopersonality_abm/markov_vocab/`
配下に`tft.txt`・`gra.txt`・`ran.txt`・`allc.txt`・`alld.txt`として
配置し、`markov_recovery.py`から読み込む小さなローダー関数を用意する
(いただいたファイルをそのまま、1行1語でコミット)。

### 3.2 テスト計画

1. **参照実装との出力一致**: 合成データ(手で作った小さな対戦ログ、または
   `markov3_widest_path_reference.py`をそのまま実行した結果)を本体の
   `build_edges`/`widest_path_threshold`に通し、参照実装の出力と
   完全一致することを確認。
2. **`series_from_moves`の変換**: 既知の手の系列(例: 全ラウンドCC)から
   期待される3-gram文字列("和和和,和和和,...,和和和。"のような)が
   出ることを確認。奇数/偶数ラウンド数、短すぎる系列(2ラウンド以下、
   3-gramが作れない)の境界も確認。
3. **手で計算できる小さな例でのwidest path**: 3〜4ノード・既知の重みの
   小さなネットワークを手で作り、widest pathのしきい値を手計算し、
   関数の出力と一致することを確認。
4. **START/GOALが無いときにNoneになること**: 「底底底」や「和和和」が
   一度も現れない参加者集合を与えて`None`が返ることを確認。
5. 人のデータは入れないので、`EXPECTED_RECOVERY`等の定数値そのものを
   使った「論文の値と一致するか」のテストはできない
   (2026-10-15に別途確認済みとのことなので、ここでは不要と判断)。

### 3.3 仮想参加者への適用(この段階では設計のみ、実装は2節のプローブの後)

- 各時点・各相手について、2節で得た「遺伝子型ごとの対戦系列」から、
  **人の群と同じ3通りの人数(56・46・69)**で、頻度に比例した重み付き
  抽出(遺伝子型の頻度が高いほど、その遺伝子型の対戦記録が抽出されやすい)
  を行い、`series_from_moves`→`build_edges`→`widest_path_threshold`で
  回復(底底底→和和和)と転落(和和和→底底底)のしきい値を計算する。
  抽出の母集団は、該当時点の該当相手との対戦系列全体(遺伝子型×30反復)。
- **抽出の繰り返し回数**: 1000回を提案(ブートストラップ的な分布を見るのに
  十分な回数で、3.4節の計算量からも問題ない規模)。
- 結果は、人の値(`EXPECTED_RECOVERY`・`EXPECTED_DESCENT_TFT`)を横線として
  重ねた分布図(ヒストグラムまたはバイオリン)で示す。TFT・GRA・RANの
  3相手×進化の時点、という構成を想定。
- 「底底底」または「和和和」が一度も現れず、widest pathが`None`になる
  (未定義になる)割合を、時点ごと・相手ごとに報告する(抽出1000回のうち
  何回がNoneだったか)。

### 3.4 拡張セットへの適用

Gradual(なだめあり)・WSLSは、予約語を64語(制限なし)として、3.3節と
同じ分析をかける(探索的)。人の実験に対応する値が無いので、横線の
重ね描きはせず、分布そのものを記述する。

---

## 4. 出力ファイル

### 4.1 進化本体(1節)

- `seed_X_generation_stats.csv`: 世代ごとの集計(世代, mean/std/min/max
  memory_depth, mean_payoff) — 既存`GenerationStats`に準拠、コミット対象
  (軽量)。
- `seed_X_genotype_census.csv.gz`: 50世代おきの「プローブ時点」ごとの
  ユニーク遺伝子型一覧(memory_depth, table, prehistoryの文字列表現,
  頻度)。ローカルのみ(gitignore、scent章の個体ログと同じ扱い)。

### 4.2 プローブ(2節)

- `seed_X_probe_gen_G_moves.csv.gz`: 生の手の系列。列は
  `genotype_id, opponent, repeat, round, own_move, opp_move`
  (21回版・100回版は`condition`列で区別、または別ファイル — 2.4節の
  確認事項(4)の結論次第)。ローカルのみ(量が多いため)。
- `seed_X_probe_gen_G_summary.csv`: 遺伝子型×相手ごとの頻度・協力率などの
  集計。コミット対象。

### 4.3 マルコフ分析(3節、プローブ完了後)

- `seed_X_markov_thresholds.csv`: 時点×相手×抽出回ごとの回復/転落しきい値
  (Noneを含む)。
- 人の値重ね描き図(PNG)、未定義割合の表。

---

## 5. この段階でやらないこと(確認済み)

- 3ステップマルコフ分析を**仮想参加者の実データに**かけること(3.3節は
  設計のみ、2節のプローブが実行されてから)。
- 事前登録の本文。
- 進化本体・プローブの本番実行(パイロットも含め、確認後に着手)。

---

## 6. ご確認いただきたい点(まとめ)

1. ノイズ率(実行誤り確率)の値 — 1%を仮提案。
2. プローブ間隔50世代でよいか(パイロット後に調整する前提でよいか)。
3. Gradualの報復開始タイミング(検知した今回の手から報復が始まる、という
   提案でよいか)。
4. 21回版・100回版を独立対戦として両方保存するか、1回100ラウンドの対戦から
   先頭20回を主分析に使う一本化にするか(計算量的には一本化を推奨)。
5. パイロット(10 seed×2000世代)→本番(50 seed×5000世代、パイロット次第で
   調整)という2段構えでよいか。
6. `population.py`の`run_duel`にノイズ引数を追加する方針(既存呼び出しは
   無変更)でよいか、それとも新しい関数として分離した方がよいか。
