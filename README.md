# EvoPersonality_ABM

進化的パーソナリティABM(エージェントベースモデル) ― Lindgren(1991)の記憶進化に、
結合強度K(自己アンカーK_self・自動同調K_out)の個体差を進化変数として重ね、
愛着スタイル(安定・拒絶・とらわれ・恐れ)に対応する4つの力学署名が集団中にどう
分布するかを、Markov構造(手の遷移ネットワーク)を指標にシミュレーションで調べる
プロジェクトです。

人間行動進化学会2026(HBES2026)での発表、およびプレプリント・投稿論文の基盤となる
シミュレーションコードと分析スクリプトを収めています。モデル設計の経緯・核となる
主張は `HBES2026_ABM骨子_二成分K.md` を参照してください。

## 人間の被験者データについて

**このリポジトリに人間の被験者ごとの記録(手の系列、質問紙の得点、ID等)は含まれて
いません。** 比較対象として使う `docs/human_network_weights.csv` は、愛着スタイル
群ごとに集計したMarkov遷移のJaccard類似度(列: `opponent,group,n,from,to,jaccard`。
`n`は各群のサンプルサイズ)のみで、個人を特定できる値・識別子は含みません。公開前の
全履歴監査の結果は `docs/PUBLIC_RELEASE_AUDIT.md` を参照してください。

## 論文・事前登録

- 論文: <!-- TODO: 投稿誌・DOIが決まり次第記入 -->
- プレプリント: <!-- TODO: PsyArXiv等のURLを決まり次第記入 -->
- 事前登録(Kモデル・Markov構造の確証的分析): `docs/PREREGISTRATION_k_model_markov.md`
- データ・コードのアーカイブ(Zenodo DOI): <!-- TODO: Zenodo公開後に記入 -->

## 再現方法

### 必要環境

- Python 3.11系を推奨(3.10以上で動作する見込み)
- git

### セットアップ

```bash
git clone https://github.com/dpforest610/EvoPersonality_ABM.git
cd EvoPersonality_ABM
python3.11 -m venv venv
source venv/bin/activate   # Windowsは venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

### テスト

```bash
pytest
```

### 主な実行順序

本番のシミュレーション結果(seedごとの生データ: census/moves/recoveryのCSV、
per-age pickleキャッシュ)はサイズが大きいため、このリポジトリには集計済みの
CSV/JSON/PNGのみを収めています。生データから再現する場合は、おおむね以下の順に
実行します(具体的な引数・所要時間の見積もりは各手順書を参照):

1. **本番ラン**: Kモデルのseedごとのシミュレーションを実行するスクリプト
   (`scripts/run_coupling_seed_sweep_50.py` など)。
2. **しきい値の計算**: `scripts/compute_k_model_type_thresholds.py` で、
   K_self・K_outの中央値分割しきい値を計算する。
3. **確証的な分析**: `scripts/run_k_model_confirmatory_analysis.py` で、
   事前登録(`docs/PREREGISTRATION_k_model_markov.md`)に従った仮説検定
   (seedブートストラップ・置換検定)を実行する。
4. **図の生成**: `scripts/plot_k_model_*.py` 系のスクリプトで、論文・学会発表用の
   図を生成する。

Macでの詳しい実行手順(コピペ可能なコマンド付き)は
`docs/K_MODEL_MARKOV_MAC_RUNBOOK.md` を参照してください。

## ライセンス

- **コード**(`evopersonality_abm/`、`scripts/`、`tests/`): MITライセンス
  (`LICENSE`参照)。
- **文書・データ**(`docs/`以下の文書、集計済みCSV/JSON、図): CC BY 4.0
  (`LICENSE-DOCS`参照)。

## 引用

このコード・データを利用する場合は `CITATION.cff` を参照して引用してください。
