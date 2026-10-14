# N=1000 発表用アニメーション(フェーズ単位の記録): 手順書

本番(N=1000)の seed 0 について、M0(none)・M1(random, p_move=0.501262)・
M2(scent, tau=0.033333)の3条件を、本番と同じパラメータ・同じ seed で
フル5000世代走らせ直し、gen 0–19 と gen 4980–4999 の2区間だけ
「1世代=5フェーズ」のフェーズ単位で位置・移動フラグ・得点を記録する。
記録は乱数の消費を変えない(オプトイン・副作用のみ)ので、この再実行は
本番そのものと同じ軌跡になるはずであり、その一致を機械的に検証できる。

実装: `evopersonality_abm/scent_spatial.py`(`phase_callback`が4つ目に
`payoff_so_far`を渡すよう拡張)、`scripts/run_scent_spatial_sweep_n1000.py`
(`--out-root`で出力先を本番ツリーの外に強制できるよう拡張。本番ルートと
同じパスを指定するとエラーで止まる)。

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
指定できる(今回の2区間用に拡張済み)。`--workers 1`(seed が1つしかないので
増やしても意味がない。3条件を同時に動かしても、worker プロセスは合計3つで
コア数を大きく下回るので安全)。

    mkdir -p logs
    nohup caffeinate -dimsu python3 scripts/run_scent_spatial_sweep_n1000.py --movement none --generations 5000 --n-seeds 1 --workers 1 --no-push --out-root scripts/output/scent_spatial_sweep_n1000_phase_recording --record-phases-seed 0 --record-phases-gens "0-19,4980-4999" > logs/n1000_phaserec_none.log 2>&1 &

    nohup caffeinate -dimsu python3 scripts/run_scent_spatial_sweep_n1000.py --movement random --p-move 0.501262 --generations 5000 --n-seeds 1 --workers 1 --no-push --out-root scripts/output/scent_spatial_sweep_n1000_phase_recording --record-phases-seed 0 --record-phases-gens "0-19,4980-4999" > logs/n1000_phaserec_random.log 2>&1 &

    nohup caffeinate -dimsu python3 scripts/run_scent_spatial_sweep_n1000.py --movement scent --tau 0.033333 --generations 5000 --n-seeds 1 --workers 1 --no-push --out-root scripts/output/scent_spatial_sweep_n1000_phase_recording --record-phases-seed 0 --record-phases-gens "0-19,4980-4999" > logs/n1000_phaserec_scent.log 2>&1 &

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

### 所要時間の見積もり

このクラウド環境(1コア)で実測: scent(M2) 約1.04秒/世代、
random(M1) 約0.83秒/世代、none(M0) 約0.90秒/世代
(`evopersonality_abm.scent_spatial.run_coupled_evolution_scent`を
population_size=1000, grid 41×41, 10世代で計測)。5000世代に換算すると
**条件あたり約70〜87分**。3条件を同時に起動すればコア数に余裕がある限り
ほぼ並列に進むので、**全体でもおよそ1.5〜2時間程度**になる見込み。

Macの実際のCPU速度はこの環境と異なるはずなので、本番を始める前に
短い計測を1回だけ推奨する(1分弱で終わる):

    time python3 scripts/run_scent_spatial_sweep_n1000.py --movement scent --tau 0.033333 --generations 50 --n-seeds 1 --workers 1 --no-push --out-root scripts/output/scent_spatial_sweep_n1000_phase_recording_TIMING_CHECK
    rm -rf scripts/output/scent_spatial_sweep_n1000_phase_recording_TIMING_CHECK

出た秒数を50で割れば1世代あたりの秒数(Mac実測値)。5000倍すれば
条件あたりの所要時間になる。

---

## 2. 再現性の検証

記録run が実際に本番の軌跡と一致するかを、記録run自身が書き出す
`seed_0_individual_log.csv.gz`(フェーズ記録とは別に、本番と同じ
100世代おきの通常スナップショットも必ず書き出される)と、
コミット済みの本番の `seed_0_individual_log.csv.gz` を突き合わせて検証する。
位置・K_out・K_self・得点を、重なる世代(gen 0, 100, ..., 4900 のうち
記録runが実際にカバーした範囲 — 5000世代フルで走らせれば50世代分すべて)
について1個体ずつ比較する。

    python3 scripts/verify_phase_recording_reproduces_production.py \
        --recording-root scripts/output/scent_spatial_sweep_n1000_phase_recording

(本番と同じMac・同じホスト名で実行する前提。違うマシンから検証する場合は
`--production-machine doeRyanoMacBook-Neo.local` を付ける。)

**このクラウド環境で、同じスクリプト・同じ出力経路で、3条件×gen 0・gen 100の
計30,000件(10 agent-fields × 1000体 × 3条件)をこの方法で既に検証済み
(不一致0件)。** 本番と同じ乱数消費であること(フェーズ記録はオプトイン・
副作用のみ)は `tests/test_scent_spatial.py::test_phase_recording_does_not_change_simulation_results`
でも保証されている。Macでの本番相当の実行では、gen 4900 を含む50地点
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

記録ファイル(`scripts/output/scent_spatial_sweep_n1000_phase_recording/**`、
条件あたりおおむね数MB、3条件合計で見積もり10〜20MB程度)は現時点で
gitignore済みでpushされない。描画は同じMac上でそのまま実行できるので、
基本は「Macで記録→Macで描画→動画ファイル(MP4/GIF、1本あたり数MB)だけ
送ってもらう」で完結するはず。動画ではなく記録ファイル自体をこちらで
描画したい場合は、ファイルサイズを見てからやり取り方法を相談する。
