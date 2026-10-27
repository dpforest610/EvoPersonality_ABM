# Kモデル Markov構造・本番結果報告

- 状態:**本番データに基づく訂正版**(初版commit`d8b6392`はH-B・H-Cの統計量の
  符号に誤りがあり、本版で訂正——下記「訂正について」を参照)
- 対象ブランチ:`claude/k-model-markov-probe`
- 事前登録:`docs/PREREGISTRATION_k_model_markov.md`(確定コミット`82203334`)
- データ:seed 200〜269(70/70完走、事前登録7節の除外規則どおり除外なし)。
  分析窓はgen 1000〜5000(プローブ間隔50、81時点)。
- 逸脱:`docs/DEVIATION_k_model_group_size.md`(2節で要約)。
- 集計元:`scripts/output/k_model_confirmatory_production/k_model_confirmatory_summary.json`
  (確証的分析、訂正後コミット`614ae04`)、`scripts/output/k_model_confirmatory_production/
  permutation_null_center.csv`(3節の探索的分析、訂正後コミット`614ae04`)。

**本報告は確証的な結果(事前登録4〜5節)と、事前登録外の探索的な結果を
はっきり分けて記載する。4節は事実の記述のみで、解釈・考察はまだ書いていない。**

## 訂正について(`docs/DEVIATION_k_model_group_size.md`7節)

確証的分析のコード(`scripts/run_k_model_confirmatory_analysis.py`)は、H-B・H-Cの差を
「戻れた割合(n_recovered/n)」のK-恐れ型−K-安定型で計算しながら、それを事前登録の
統計量「戻れなかった割合(never_recovered_rate)」のK-恐れ型−K-安定型として扱っていた
(符号がちょうど逆)。この誤りはコードの不具合であり、事前登録からの逸脱ではない。
最初に出力された`k_model_confirmatory_summary.json`(commit`cab6ae92`)のH-B・H-Cの
判定(H-B支持、H-C部分的に支持)は、この誤りにより、本来は支持されない結果を
「支持」と判定していた。コードを修正し(`never_recovered_rate`の差に直す)、修正前に
書かれた時点ごとのキャッシュはその場で符号を反転して読み直す形で、計算をやり直さず
要約を作り直した(`normalize_cached_result`)。**H-AとH-B・H-Cの件数(n・n_recovered)
自体は、最初から影響を受けていない。** 本報告はこの訂正後の値(commit`614ae04`)を使う。

## p値・判定の書き方について

**seedブートストラップ**(事前登録5-2節、70 seedを復元抽出で2000回選び直す)の
片側p値は`p = (差が0以下になった回数 + 1) / 2001`(0除算を避ける+1補正)。
2000回のブートストラップのうち1回も「差が0以下」にならなかった場合は
`p < 1/2001`、全回「差が0以下」だった場合は`p > 2000/2001`と表記し、それ以外は
具体的な値(`p = .xxxx`)を示す(`docs/MARKOV_STRUCTURE_EVOLUTION_RESULTS.md`と
同じ規約)。

**置換検定**(事前登録5-4節、1000回の置換)の経験的p値は
`p = (帰無分布がtrue以上になった回数 + 1) / 1001`。同じ規約で、最小値は
`p < 1/1001`、最大値は`p > 1000/1001`と表記する。

**最終判定**(事前登録5-4節step 6の規則):各比較について、**seedブートストラップが
Holm補正後に有意(α=.05)、かつ置換検定もp<.05の、両方が成り立つときだけ**
「型による違い」として扱う。一方だけが有意な場合は、その比較は支持されないものと
して扱う。各仮説の判定は、含まれる比較がすべて「両方有意」のとき「支持」、一部のみ
のとき「部分的に支持」、どれも「両方有意」でないとき「支持されない」とする
(事前登録5-3節の判定基準をそのまま使う)。

---

## 1. 確証的結果(事前登録4〜5節、主分析:系統間抽出、gen 1000〜5000平均)

### H-A1(K-安定型↔Secure):r(K-安定型,Secure) が r(K-安定型,Mix)・r(K-安定型,Fearful) より高いか

| 比較 | 点推定の差 | seedブートストラップ | Holm | 置換検定 | 両方有意 |
|---|---|---|---|---|---|
| r(K_Secure,Secure)-r(K_Secure,Mix) | -0.0425 | p > 2000/2001 | × | p > 1000/1001 | × |
| r(K_Secure,Secure)-r(K_Secure,Fearful) | -0.0177 | p > 2000/2001 | × | p < 1/1001 | × |

**判定:支持されない**(両比較とも点推定の差の符号自体が予想と逆——K-安定型の網は
Secureより他の2群の網に近い)。

### H-A2(合成Mix↔Mix):r(Mix,Mix) が r(Mix,Secure)・r(Mix,Fearful) より高いか

| 比較 | 点推定の差 | seedブートストラップ | Holm | 置換検定 | 両方有意 |
|---|---|---|---|---|---|
| r(Mix,Mix)-r(Mix,Secure) | +0.0317 | p < 1/2001 | ○ | p < 1/1001 | ○ |
| r(Mix,Mix)-r(Mix,Fearful) | +0.0112 | p = .0010 | ○ | p < 1/1001 | ○ |

**判定:支持**

### H-A3(K-恐れ型↔Fearful):r(K-恐れ型,Fearful) が r(K-恐れ型,Secure)・r(K-恐れ型,Mix) より高いか

| 比較 | 点推定の差 | seedブートストラップ | Holm | 置換検定 | 両方有意 |
|---|---|---|---|---|---|
| r(K_Fearful,Fearful)-r(K_Fearful,Secure) | +0.0365 | p < 1/2001 | ○ | p < 1/1001 | ○ |
| r(K_Fearful,Fearful)-r(K_Fearful,Mix) | +0.0250 | p < 1/2001 | ○ | p < 1/1001 | ○ |

**判定:支持**

### H-B(回復):never_recovered_rate が K-安定型 < K-恐れ型 か(SELF・POP)【訂正後】

| 比較 | 点推定の差 | seedブートストラップ | Holm | 置換検定 | 両方有意 |
|---|---|---|---|---|---|
| H-B1(SELF):never_recovered_rate(K_Fearful,SELF)-never_recovered_rate(K_Secure,SELF) | -0.0913 | p > 2000/2001 | × | p > 1000/1001 | × |
| H-B2(POP):never_recovered_rate(K_Fearful,POP)-never_recovered_rate(K_Secure,POP) | -0.0849 | p = .9995 | × | p > 1000/1001 | × |

**判定:H-B1(SELF)支持されない、H-B2(POP)支持されない**(両比較とも点推定の差の
符号が予想と逆——never_recovered_rateはK-恐れ型のほうがK-安定型より低い)。

### H-C1(記憶の深さの層):H-B1の差は、実効記憶深さ(depthLow 0〜2 / depthMid 3 / depthHigh 4以上)の
どの層でも残るか(SELF)【訂正後】

最低人数(20、事前登録4節で確定)を下回る層はなかった(`untestable_strata: []`、3層とも検定可能)。

| 層 | 点推定の差 | seedブートストラップ | Holm | 置換検定 | 両方有意 |
|---|---|---|---|---|---|
| depthLow(0〜2) | +0.0152 | p = .2459 | × | p > 1000/1001 | × |
| depthMid(3) | -0.1263 | p > 2000/2001 | × | p > 1000/1001 | × |
| depthHigh(4以上) | -0.1465 | p > 2000/2001 | × | p > 1000/1001 | × |

**判定:支持されない**(3層とも「両方有意」にならない。depthMid・depthHighは
点推定の差の符号も予想と逆)。

### まとめ(事実のみ、解釈は書かない)

| 仮説 | 判定 |
|---|---|
| H-A1(K-安定型↔Secure) | 支持されない |
| H-A2(合成Mix↔Mix) | 支持 |
| H-A3(K-恐れ型↔Fearful) | 支持 |
| H-B1(SELF) | 支持されない |
| H-B2(POP) | 支持されない |
| H-C1(記憶の深さ) | 支持されない |

---

## 2. 逸脱(事前登録3節「群の作り方」)

`docs/DEVIATION_k_model_group_size.md`の要約:

- 事前登録は、系統間抽出で各群を「人の群の人数と同じ別々のseedから1体ずつ」選ぶとしていた
  (K-安定型56・合成Mix46・K-恐れ型69)。
- 本番データ(81時点)で、その類型の個体が1体以上いるseedの数を数えたところ、
  **K-恐れ型は81時点中60時点で69 seedに届かず**(最少63 seed)、事前登録どおりの抽出は
  実行不能だった(K-安定型・合成Mixは0/81時点で不足なし)。
- 2026-10-26、安念が決定(**確証的分析の結果を一つも見る前**、census のK の値のみで決定):
  いる seed すべてから1体ずつ選び、群の人数 = min(人の群の人数, その類型がいるseedの数)
  とする。系統間抽出の考え方(1系統から1体)は保つ。この規則を5-1節(点推定)・5-2節
  (seedブートストラップ)・5-4節(置換検定、本物のラベルの統計量も含む)に一貫して適用した。
- 各時点・各群で実際に使った人数は`ha_effective_group_size`として記録済み(本番データの
  `_age_cache`、Mac local)。

**K-恐れ型の群の人数の範囲:63〜69**(69 seed揃った21/81時点では69、残り60/81時点では
その時点でK-恐れ型の個体がいたseedの数[最少63])。K-安定型は56、合成Mixは46で、
どちらも全81時点で変動なし(不足がなかったため)。

(1節のH-B・H-Cの符号訂正は、上の人数の規則とは別の問題——コードの不具合であり、
事前登録からの逸脱ではない。「訂正について」節を参照。)

---

## 3. 探索的な記述(事前登録外と明記、解釈はまだ書かない)

置換検定(事前登録5-4節)は、各(seed, 時点)の**中で**類型ラベルを入れ替える。そのため、
帰無分布は必ずしも0を中心にしない——ラベルを入れ替えても、どの系統(seed)がどれだけ
その置換のサンプルに寄与するかという系統間の違い自体は残るため。帰無分布の中心
(平均)と、本物の差との差(「型そのものの分」)を別途計算すれば、本物の差の見かけの
符号・大きさが、系統間の違いだけでどこまで説明されるかを記述できる。

計算は`scripts/analyze_k_model_permutation_null_center.py`で行った
(`scripts/output/k_model_confirmatory_production/permutation_null_center.csv`、
訂正後コミット`614ae04`)。

| 仮説 | 比較 | 本物の差(true) | 置換の中心(null_center) | その差(type_only) |
|---|---|---|---|---|
| H-A1 | r(K_Secure,Secure)-r(K_Secure,Mix) | -0.0419 | -0.0305 | -0.0114 |
| H-A1 | r(K_Secure,Secure)-r(K_Secure,Fearful) | -0.0172 | -0.0216 | +0.0044 |
| H-A2 | r(Mix,Mix)-r(Mix,Secure) | +0.0311 | +0.0294 | +0.0017 |
| H-A2 | r(Mix,Mix)-r(Mix,Fearful) | +0.0106 | +0.0083 | +0.0023 |
| H-A3 | r(K_Fearful,Fearful)-r(K_Fearful,Secure) | +0.0363 | +0.0210 | +0.0153 |
| H-A3 | r(K_Fearful,Fearful)-r(K_Fearful,Mix) | +0.0245 | -0.0077 | +0.0321 |
| H-B1 | never_recovered_rate(K_Fearful,SELF)-(K_Secure,SELF) | -0.0913 | -0.0320 | -0.0593 |
| H-B2 | never_recovered_rate(K_Fearful,POP)-(K_Secure,POP) | -0.0849 | -0.0467 | -0.0382 |
| H-C1 | depthLow | +0.0152 | +0.0501 | -0.0349 |
| H-C1 | depthMid | -0.1263 | -0.0508 | -0.0754 |
| H-C1 | depthHigh | -0.1465 | -0.0817 | -0.0648 |

(「本物の差」列は`k_model_confirmatory_summary.json`の`permutation`配列の
`true_diff_window_avg`と一致している——1節の点推定と、丸め誤差の範囲で一致することを
確認済み。)
