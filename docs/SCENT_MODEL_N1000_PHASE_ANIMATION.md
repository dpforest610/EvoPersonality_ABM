# N=1000 発表用アニメーション(フェーズ記録+毎世代記録): 手順書

本番(N=1000)の seed 0 について、M0(none)・M1(random, p_move=0.501262)・
M2(scent, tau=0.033333)の3条件を、本番と同じパラメータ・同じ seed で
フル5000世代走らせ直す。1回の再実行の中で2種類の記録を同時に取る:

- **フェーズ記録**: gen 0–19 と gen 4980–4999 の2区間だけ、
  「1世代=5フェーズ」のフェーズ単位で位置・移動フラグ・得点を記録
  (`phase_recording.pkl.gz`)。
- **毎世代記録(追加A)**: gen 0–4999 の全世代で、本番の individual_log
  と同じ列(pos_x, pos_y, k_self, k_out, payoff ほか)を記録
  (`seed_0_individual_log.csv.gz`。中身は本番と同じ形式で、100世代おき
  ではなく毎世代分が入っている版)。

記録は乱数の消費を変えない(オプトイン・副作用のみ)ので、この再実行は
本番そのものと同じ軌跡になるはずであり、その一致を機械的に検証できる。

この2種類の記録から、3つの成果物を作る: (1)フェーズ単位の3パネル動画
(3節、既にデザイン確認済み)、(2)追加B: Heatbugs風4タイプ動画・毎世代版
(5節、記録完了後にMacで描画)、(3)追加C: 学会用の集計図(6節、**本番データ
だけで作れるので、このクラウド環境で既に作成済み**)。

実装: `evopersonality_abm/scent_spatial.py`(`phase_callback`が4つ目に
`payoff_so_far`を渡すよう拡張)、`scripts/run_scent_spatial_sweep_n1000.py`
(`--out-root`で出力先を本番ツリーの外に強制できるよう拡張。本番ルートと
同じパスを指定するとエラーで止まる。`--individual-log-interval`で
individual_log の記録間隔を既定の100世代おきから1世代おきに変えられる
よう拡張 -- `seed_summary.csv`を作る定常状態スナップショットの間隔
[既定どおり100世代おき・gen>=2000]には影響しない、別の設定)。

---

## 0. 前提

- 本番を動かしたのと同じMacの、同じリポジトリのclone(既存の venv・
  コミット済みの本番データがあるはず)。
- 新しいコード(この手順の元になった変更)を pull してあること:

      cd ~/path/to/EvoPersonality_ABM   # 本番を動かしたディレクトリ
      git pull origin claude/lindgren-population-evolution-08fat1
      python3 -m pytest -q   # 162 件 pass するはず

- 出力先は本番ディレクトリ(`scripts/output/scent_spatial_sweep_n1000/`)
  とは別の `scripts/output/scent_spatial_sweep_n1000_phase_recording/` に
  固定する。`--out-root` に本番のパスそのものを渡すとスクリプトが
  エラーで停止する(`scripts/run_scent_spatial_sweep_n1000.py`の
  `resolve_experiment_root`、`tests/test_run_scent_spatial_sweep_n1000.py`
  の `test_resolve_experiment_root_rejects_the_production_root_itself`
  で保証)。このディレクトリは丸ごと `.gitignore` 済み(個体ログも
  フェーズ記録もチェックポイントも、何もpushされない)。

---

## 1. 記録用の実行

3条件、それぞれ1行。`--record-phases-gens` はカンマ区切りで複数区間を
指定できる(今回の2区間用に拡張済み)。`--individual-log-interval 1`を
付けると、フェーズ記録と同じ実行の中で、全世代分の individual_log も
書き出される(追加A)。`--workers 1`(seed が1つしかないので増やしても
意味がない。3条件を同時に動かしても、worker プロセスは合計3つでコア数を
大きく下回るので安全)。`--chunk-size 1000`(既定の250より大きくした。
追加Aでは、未完了のチェックポイントに世代ごとの行がすべて溜まっていくので
(5000世代の終わり近くでは最大約500万行)、チェックポイントの保存間隔が
短すぎると保存のたびに時間がかかる。1000なら5回で済む。長めの間隔でも、
Ctrl-C やスリープで止まった場合に最大1000世代分ぶん計算をやり直すだけで、
安全性は保たれる)。

    mkdir -p logs
    nohup caffeinate -dimsu python3 scripts/run_scent_spatial_sweep_n1000.py --movement none --generations 5000 --n-seeds 1 --workers 1 --no-push --chunk-size 1000 --out-root scripts/output/scent_spatial_sweep_n1000_phase_recording --record-phases-seed 0 --record-phases-gens "0-19,4980-4999" --individual-log-interval 1 > logs/n1000_phaserec_none.log 2>&1 &

    nohup caffeinate -dimsu python3 scripts/run_scent_spatial_sweep_n1000.py --movement random --p-move 0.501262 --generations 5000 --n-seeds 1 --workers 1 --no-push --chunk-size 1000 --out-root scripts/output/scent_spatial_sweep_n1000_phase_recording --record-phases-seed 0 --record-phases-gens "0-19,4980-4999" --individual-log-interval 1 > logs/n1000_phaserec_random.log 2>&1 &

    nohup caffeinate -dimsu python3 scripts/run_scent_spatial_sweep_n1000.py --movement scent --tau 0.033333 --generations 5000 --n-seeds 1 --workers 1 --no-push --chunk-size 1000 --out-root scripts/output/scent_spatial_sweep_n1000_phase_recording --record-phases-seed 0 --record-phases-gens "0-19,4980-4999" --individual-log-interval 1 > logs/n1000_phaserec_scent.log 2>&1 &

3つまとめて同時に起動してよい(3プロセス、各1 worker)。`--generations 5000`を
省略しても既定値は5000だが、明示しておく。`--push-checkpoints`は付けない
(`--no-push`だけで、このスクリプトのgit関連処理は何も起動しない)。

### 確認・停止・再開

    # 進捗(1秒あたりの世代数、チャンクごとの経過時間)
    tail -f logs/n1000_phaserec_scent.log

    # 動いているか
    ps aux | grep run_scent_spatial_sweep_n1000 | grep -v grep

    # 完了したか(各条件、seed_0_summary.csv ができれば完了。マシン名の
    # サブディレクトリ名は本番と同じはず -- 本番の
    # scripts/output/scent_spatial_sweep_n1000/none/<マシン名>/ と同じ名前)
    find scripts/output/scent_spatial_sweep_n1000_phase_recording -name seed_0_summary.csv

途中で止めても(Ctrl-C やターミナルを閉じる、Macがスリープするなど)
チェックポイント(`seed_0_checkpoint.pkl`)から自動で再開できる。同じ
コマンドをもう一度実行するだけでよい。

### 所要時間とファイルサイズの見積もり

このクラウド環境(1コア)で実測: scent(M2) 約0.9〜1.0秒/世代、
random(M1) 約0.83秒/世代、none(M0) 約0.90秒/世代。
`--individual-log-interval 1`(毎世代記録)を付けても、この程度の規模
(30世代で実測)では1世代あたりの時間は変わらなかった。5000世代に
換算すると**条件あたり約70〜87分**。3条件を同時に起動すればコア数に
余裕がある限りほぼ並列に進むので、**全体でもおよそ1.5〜2時間程度**に
なる見込み(フェーズ記録のみの場合と同程度)。

ファイルサイズ: `seed_0_individual_log.csv.gz` は、毎世代記録(追加A)で
約0.0223 MB/世代(このクラウド環境で scent・30世代・population=1000で実測)
なので、**5000世代で条件あたり約110 MB**の見込み(本番の50時点・約1.1MB/条件
から単純に100倍しても同程度の約110MB -- 同じオーダー)。3条件合計で
約330MB程度。`phase_recording.pkl.gz`は従来どおり条件あたり数MB。

Macの実際のCPU速度・ディスク速度はこの環境と異なるはずなので、本番を
始める前に短い計測を1回だけ推奨する(1分弱で終わる):

    time python3 scripts/run_scent_spatial_sweep_n1000.py --movement scent --tau 0.033333 --generations 50 --n-seeds 1 --workers 1 --no-push --out-root scripts/output/scent_spatial_sweep_n1000_phase_recording_TIMING_CHECK --individual-log-interval 1
    du -h scripts/output/scent_spatial_sweep_n1000_phase_recording_TIMING_CHECK/scent/*/seed_0_individual_log.csv.gz
    rm -rf scripts/output/scent_spatial_sweep_n1000_phase_recording_TIMING_CHECK

出た秒数を50で割れば1世代あたりの秒数、ファイルサイズを50で割れば
1世代あたりのサイズ(どちらもMac実測値)。5000倍すれば条件あたりの
所要時間・ファイルサイズになる。

---

## 2. 再現性の検証

記録run が実際に本番の軌跡と一致するかを、記録run自身が書き出す
`seed_0_individual_log.csv.gz`(追加Aにより、毎世代分 gen 0–4999 すべて
入っている)と、コミット済みの本番の `seed_0_individual_log.csv.gz`
(100世代おき、gen 0, 100, ..., 4900 の50時点)を突き合わせて検証する。
スクリプト側は重なる世代だけを自動で選んで比較するので、追加Aにより
**50時点すべて**(以前のフェーズ記録だけの場合はgen 0・100の2時点のみ)
が対象になる -- コマンド自体は変更なし:

    python3 scripts/verify_phase_recording_reproduces_production.py \
        --recording-root scripts/output/scent_spatial_sweep_n1000_phase_recording

(本番と同じMac・同じホスト名で実行する前提。違うマシンから検証する場合は
`--production-machine doeRyanoMacBook-Neo.local` を付ける。)

位置・K_out・K_self・得点を、重なる世代それぞれについて1個体ずつ比較する
(50時点 × 1000体 × 5項目 × 3条件 = 計750,000件の比較になる見込み)。

**このクラウド環境で、同じスクリプト・同じ出力経路で、追加Aを使って
3条件×gen 0のみ(毎世代記録のクイック動作確認、計5,000件)と、
フェーズ記録のみの3条件×gen 0・gen 100(計30,000件)の両方をこの方法で
既に検証済み(不一致0件)。** 本番と同じ乱数消費であること(フェーズ記録・
毎世代記録ともオプトイン・副作用のみ)は
`tests/test_scent_spatial.py::test_phase_recording_does_not_change_simulation_results`
と `tests/test_run_scent_spatial_sweep_n1000.py::test_individual_log_interval_never_changes_seed_summary`
でも保証されている。Macでの本番相当の実行(5000世代フル)では、50時点
すべてで同様の一致を確認できるはずで、そのレポート(不一致件数)を
そのまま「本番そのものの映像である」根拠として使える。

gen 4980–4999 そのものは本番の個体ログに記録がない(100世代おきのため)
ので直接の突き合わせはできないが、乱数消費が変わらないこと(上記テスト)
と、gen 0・100・...・4900 での完全一致が確認できれば、同じ仕組みで
記録された gen 4980–4999 も本番と同一の軌跡であると言える。

不一致が出た場合は、report に差分(世代・個体・項目・両方の値)が
そのまま出力されるので、そのログを送ってもらえれば原因を調べる。

---

## 3. 描画

記録ファイル(`phase_recording.pkl.gz`、1条件あたり数MB程度)だけを読む。
本番の個体ログやGitHubへのアクセスは不要なので、Macでもこのクラウドの
checkoutでも同じコマンドで動く。2本(早期/後期)をそれぞれ生成:

    python3 scripts/animate_scent_n1000_phase_recording.py \
        --recording-root scripts/output/scent_spatial_sweep_n1000_phase_recording \
        --gen-range 0-19 --out-name early

    python3 scripts/animate_scent_n1000_phase_recording.py \
        --recording-root scripts/output/scent_spatial_sweep_n1000_phase_recording \
        --gen-range 4980-4999 --out-name late

出力: `scripts/output/figures/scent_n1000_phase_animation_{early,late}.{mp4,gif}`。
1本あたりの描画時間は、このクラウド環境で6世代分(102フレーム)の
試し描画が約1分40秒だったので、20世代分(約340フレーム)なら
5〜6分程度の見込み(MP4・GIF各1本、合計で条件×レンジ1本あたり)。

デザイン(すでにこのクラウド環境で試し描画・確認済み、プレビューを送付済み):
- 3パネル横並び(M0/M1/M2)、同じフェーズで同期。16:9、白背景、英語表記。
- 1コマ=1フェーズ。移動は4コマで補間(フェーズ間の遷移のみ;
  Kや得点・背景の匂いの場は補間せず、遷移先フェーズの値をそのまま表示)。
  トーラスの端をまたぐ移動は最短経路(`torus_delta`)で描く。
- 世代の境界(子が親を置き換える)はフェーズ間の補間をせず、ハードカット
  (`evopersonality_abm/scent_spatial.py`の設計どおり、agent_indexは
  世代をまたいで同一個体を指さないため)。
- 点の色=K_out(viridis)、点の大きさ=K_self、その遷移で動いた個体は
  白い縁取りで目立たせる。
- 背景=匂いの場: 各マスについて、占有されているMoore近傍(radius=1、
  `_mean_neighbor_payoff`と同じ定義)の1回あたり得点の平均。Redsの
  グラデーション、占有された近傍が1つもないマスは白。色の範囲は、その
  動画(early/late)の全コマ共通で固定(両方のレンジを通した固定ではない)。
- 上部に"Generation g, phase p/5"、各パネル下に「そのフェーズの移動率」。

N=1000・41×41グリッドはN=100・13×13版より密度は同じだが点の総数が
10倍なので、点のサイズはN=100版のパイロット動画より大幅に小さくして
見やすさを確保した(`size_from_kself`)。

---

## 4. 受け渡し

記録ファイル(`scripts/output/scent_spatial_sweep_n1000_phase_recording/**`)
は現時点で丸ごとgitignore済みでpushされない。内訳の見積もり:
`phase_recording.pkl.gz`は条件あたり数MB、`seed_0_individual_log.csv.gz`
は追加A込みで条件あたり約110MB(3条件合計で約330MB)。

描画(`animate_scent_n1000_phase_recording.py`)は`phase_recording.pkl.gz`
だけを読むので、Macで記録→Macで描画→動画ファイル(MP4/GIF、1本あたり
数MB)だけ送ってもらう形で完結するはず。`seed_0_individual_log.csv.gz`
(毎世代記録)はサイズが大きいので、再現性の検証(2節)・5節(追加B)以外の
用途、たとえばこのクラウド側で別の描画を試したい場合などに必要になったら、
ファイルサイズを見てやり取り方法(圧縮して一部だけ送る、特定の条件・
世代範囲だけ抜き出すなど)を相談する。5節(追加B)の動画もMac上で
`seed_0_individual_log.csv.gz`から直接描画できるので、追加の転送は不要。

---

## 5. 追加B: Heatbugs風4タイプ動画(毎世代版)

1節の記録run(`--individual-log-interval 1`込み)が完了すれば、追加の
記録は不要。`seed_0_individual_log.csv.gz`から直接描画する。

### 固定の型分けしきい値

型の境目(K_self・K_out それぞれの中央値)は、本番の**全30 seed × 3条件
(M0・M1・M2)、gen≥2000をプールして**一度だけ計算し固定した(試作で使った
seed 0 だけの値 0.7835265 / 0.4566165 は置き換えた)。このクラウド環境で
既に計算済み:

    K_self 中央値(threshold_kself) = 0.7770470
    K_out  中央値(threshold_kout)  = 0.5025085

(`scripts/compute_scent_n1000_type_thresholds.py`で計算、
`scripts/output/scent_spatial_sweep_n1000/analysis/type_thresholds.json`
に保存済み。本番データはこのクラウド環境に既にあるコミット済みの生ログを
使っているので、Mac側で再計算する必要はない。)

### 描画コマンド

3条件 × 2速度版、計6本。`--speed full`が5000世代@60fps(約85秒)、
`--speed fast`が2世代おき@60fps(約40秒)。

    python3 scripts/animate_scent_n1000_heatbugs.py --movement none --recording-root scripts/output/scent_spatial_sweep_n1000_phase_recording --speed full
    python3 scripts/animate_scent_n1000_heatbugs.py --movement none --recording-root scripts/output/scent_spatial_sweep_n1000_phase_recording --speed fast
    python3 scripts/animate_scent_n1000_heatbugs.py --movement random --recording-root scripts/output/scent_spatial_sweep_n1000_phase_recording --speed full
    python3 scripts/animate_scent_n1000_heatbugs.py --movement random --recording-root scripts/output/scent_spatial_sweep_n1000_phase_recording --speed fast
    python3 scripts/animate_scent_n1000_heatbugs.py --movement scent --recording-root scripts/output/scent_spatial_sweep_n1000_phase_recording --speed full
    python3 scripts/animate_scent_n1000_heatbugs.py --movement scent --recording-root scripts/output/scent_spatial_sweep_n1000_phase_recording --speed fast

出力: `scripts/output/figures/scent_n1000_heatbugs_{none,random,scent}_{full,fast}.mp4`。

デザイン(添付の`heatbugs_style_reference.py`と同じ見た目。このクラウド環境で
80世代・population=1000の試し描画で確認済み):
- 左: 黒背景、空きマスは匂いの場(hotカラーマップ、範囲は`--vmin`/`--vmax`で
  固定、既定2.55〜3.15)、エージェントは4タイプの色の正方形(白=高K_self/高K_out、
  青=高K_self/低K_out、緑=低K_self/高K_out、ピンク=低K_self/低K_out)。
  匂いの場は、モデル本体(`_mean_neighbor_payoff`、radius=1のMoore近傍=8マス)
  と同じ定義に修正した(参考スクリプトは上下左右4マスの誤りだった)。
- 右上: タイプ割合(%)の推移、世代とともに線が伸びる。
- 右下: タイプごとの「このタイプの平均得点 − 全体平均得点」の累積和、
  世代とともに線が伸びる。
- 画面下に「中央値で無理やり分けた型であり、集団は連続した一つの雲」の注記。
- 英語表記、16:9、MP4(フルHD、dpi100)。

所要時間: このクラウド環境で80世代・60fps相当の設定で約30秒
(描画自体はCPU律速で世代数にほぼ比例)。5000世代版(`--speed full`)は
この約60倍の計算量になるので、1本あたり数分〜十数分程度を見込む
(3条件×2速度=6本で合計数十分程度)。実際の時間はMacの性能次第なので、
1節のタイミング計測ついでに、80〜100世代程度の短い`--individual-log-interval 1`
記録で先にこのスクリプトを試すことを推奨する。

---

## 6. 追加C: 学会用の集計図(本番データのみ、作成済み)

再実行を待たずに、**既にこのクラウド環境で作成済み**。本番の全30 seed ×
各条件(M0・M1・M2)、gen≥2000(100世代おき、30時点/seed)を使い、
5節と同じ固定しきい値(K_self≥0.7770470、K_out≥0.5025085)で型分けし、
seed間ブートストラップ(2000回)の平均+95%帯を3パネル(M0/M1/M2)で描いた:

- `docs/figures/scent_n1000/type_share_summary_{print,slide}.png`
  (タイプ割合の推移)
- `docs/figures/scent_n1000/type_cumulative_deviation_summary_{print,slide}.png`
  (タイプごとの累積得点優位)

`_print`が300dpi、`_slide`が16:9・大きめフォント版。探索的・記述的な図
として扱い、検定は付けていない。計算は
`scripts/compute_scent_n1000_type_thresholds.py` →
`scripts/plot_scent_n1000_type_summary.py`。

注意: 本番の個体ログは100世代おきのスナップショットしかないため、この
図の累積和は100世代ごとの30時点についての累積和であり、毎世代の累積和
ではない。毎世代版が必要なら、5節(追加B)と同じ`seed_0_individual_log.csv.gz`
(毎世代記録)から作り直すことになる(現時点では未実装・未作成)。
