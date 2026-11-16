# 公開前点検(dpforest610/EvoPersonality_ABM)

実施日: 2026-11-16。対象: `git log --all`(全ブランチ・全リモート追跡ブランチ)の
全履歴。本リポジトリをGitHubで公開した場合、全ブランチの全コミット履歴が
閲覧可能になることを踏まえ、現在のデフォルトブランチ(`main`)だけでなく
全履歴を対象に調べた。**この文書は点検結果の報告のみ。公開・履歴の書き換え・
ファイルの削除は一切実行していない。**

## 結論(先に短く)

- **人のデータ・秘密情報・50MB超のファイル: いずれもゼロ。この3点は公開して問題ない。**
- **公開前に対処が必要な点が2つ:**
  1. **`main`ブランチが論文の実データ・分析を一切含んでいない**(2026-08-28で止まっており、
     ごく初期の試作コードのみ)。実際の研究(Markov構造分析・Kモデル)は
     `claude/markov-production-analysis`と`claude/k-model-markov-probe`という
     2本の長期ブランチにしかない。`main`をそのまま既定ブランチとして公開すると、
     論文が記述する内容と一致しないコードだけが表に出る。
  2. **研究者のMacの合成ホスト名`doeRyanoMacBook-Neo.local`**が、コミット
     メッセージ191件と、現在のツリーにも残るファイルパス約495件に埋め込まれている。
     機密情報ではないが、個人のPCを特定できる情報として公開前に残すか消すか判断が要る。
- 整理すべき不要なブランチが8本ある(§5)。削除しても、論文が引用する8つの
  コミットはすべて保持される設計で提案している。
- 公開用の下書き(README・LICENSE・CITATION.cff・`.zenodo.json`)を
  `claude/public-release-prep`ブランチに作成した(§6)。`main`にはまだ入れていない。

---

## 1. 人のデータが入っていないか

**結論: 個人の被験者データは、全履歴のどこにも見つからなかった。**

調査方法: `git rev-list --objects --all`で全ブランチ・全時点を通じて一度でも
コミットされた全オブジェクト(36,707オブジェクト、26,906の重複なしパス)を洗い出し、
拡張子・ディレクトリ名・内容で選別した。

確認した事実:
- `.xlsx` / `.xls` / `.docx` / `.doc` / `.zip` / `.pptx` / `.pdf` は、全履歴を通じて
  **一度も**コミットされていない。
- トップレベルに存在したディレクトリは`docs/`・`evopersonality_abm/`・`progress/`・
  `scripts/`・`tests/`の4+1個のみ。`data/`・`raw/`・`human_data/`・`subjects/`の
  ようなディレクトリは一度も存在しない。
- 履歴中に存在した26,246個のCSVのうち26,233個は`scripts/output/`以下
  (ABM自身が生成した合成シミュレーション出力)。
- 履歴中の画像ファイル(約115個)はすべてmatplotlib製のグラフ・アニメーションで、
  スキャン画像・同意書・アンケート画面などではない。
- JSONファイルはすべて`scripts/output/`または`progress/`以下のシミュレーション
  メタデータ・チェックポイント。
- 氏名・学籍番号・被験者・参加者ID・個人情報・participant_id等のキーワードで
  全履歴の本文を検索したが、ヒットしたのはいずれも「個人データを含まないことを
  説明している本プロジェクト自身の文書の記述」のみだった。

### `docs/human_network_weights.csv` の内容確認

最初のコミット(`473c4feb`、2026-10-16)から現在の版(`60aaf9cd`)まで、
スキーマは一度も変わっていない。

```
opponent,group,n,from,to,jaccard
TFT,Secure,56,和和和,和和搾,0.555556
TFT,Secure,56,和和搾,和搾底,0.390244
...
```

1087行、各行は「愛着スタイル群 × 対戦相手 × 手の遷移(from→to) × その群での
Jaccard類似度」という**群単位の集計値**。`n`は個人の識別子ではなく、その群の
サンプルサイズ(46〜69人程度)。個人ごとの値・識別子は一切含まれない。
これを読み込む唯一のコード(`evopersonality_abm/markov_apply.py`の
`load_human_weights()`)も、集計値であることを前提に書かれている。
ユーザーの認識(「群ごとの集計値なので可」)の通りで問題ない。

---

## 2. 秘密の情報が入っていないか

**結論: 秘密情報は、全履歴のどこにも見つからなかった。**

- `gitleaks`(v8、`go install`で導入)を全ブランチ・全履歴に対して実行
  (`gitleaks detect --source . --log-opts="--all"`)。1748コミット・約1.35GBの
  blob内容を、既定のルールセット(AWS/GCP/Azureキー、GitHub/GitLabトークン、
  Slackトークン、PEM秘密鍵ヘッダ、汎用の高エントロピー文字列など)で走査し、
  **検出ゼロ**。
- 独立した手動の正規表現掃引(`git log --all -p`全体に対して、`ghp_`・
  `github_pat_`・`AKIA...`・PEM秘密鍵ヘッダ・`AIza...`・`sk-...`形式・
  `password=`/`secret_key=`/`api_key=`等のパターン)でも**検出ゼロ**。
- `.env`ファイルは全履歴を通じて一度もコミットされていない。
- `id_rsa`・`*.pem`・`*.pfx`・`*.p12`・`.netrc`・`known_hosts`・
  `credentials.json`・`service-account*.json`のような、認証情報を示唆する
  ファイル名も一度も追加されていない。

---

## 3. 個人に結びつく情報

### 3-1. コミットの作者・コミッターのメールアドレス(全履歴・重複なし)

| 識別子 | 役割 | コミット数 | 期間 |
|---|---|---|---|
| `Claude <noreply@anthropic.com>` | クラウド実行時のbot identity | 1145 | 2026-08-25〜2026-11-02 |
| `doeRya <doerya@doeRyanoMacBook-Neo.local>` | 研究者自身のMacでのgit identity(gitが自動生成する`user@hostname`形式。実在のメールアドレスではない) | 419 | 2026-10-12〜2026-11-02 |
| `dpforest610 <...@users.noreply.github.com>` | リポジトリ所有者のGitHub noreplyエイリアス | 217 | 2026-08-25〜2026-10-09 |

ユーザー本人の個人Gmailアドレスは、全履歴のどこにも一度も現れない。

### 3-2. ホスト名・ユーザー名・ローカルパス

| パターン | 場所 | 残っているブランチ | 件数 |
|---|---|---|---|
| `doeRyanoMacBook-Neo.local`(コミットメッセージ中) | 「...進捗スナップショット」系のチェックポイント・コミット件名 | 5ブランチ(`claude/k-model-markov-probe`含む) | 191コミット |
| 同ホスト名(**ファイル・ディレクトリパスそのもの**) | 例: `scripts/output/scent_spatial_sweep_n1000/.../doeRyanoMacBook-Neo.local/run_metadata.json` — **現在のツリーにも残っている**(過去の履歴だけではない) | 5〜6ブランチ | 495個の重複なしパス |
| 同ホスト名(JSON内容、例 `"machine": "doeRyanoMacBook-Neo.local"`) | 上と同じファイル群の中身 | 同上 | 約495箇所 |
| `/Users/<名前>/...`形式の絶対パス | 全履歴の本文(`-G`検索) | — | **0件**(このセッションの会話中で貼り付けられたターミナル出力に含まれていただけで、コミットされたことは一度もない) |
| Macのシェルプロンプト文字列(`doerya@...`等) | 現在の全ブランチの`git grep` | — | **0件** |

重要な点: `/Users/doerya/...`のようなパスやシェルプロンプトの文言は、この会話で
ユーザーが貼り付けたターミナル出力の中にだけ存在し、**リポジトリには一度も
コミットされていない**ことを確認した。実際に残っているのは、
「`doeRyanoMacBook-Neo.local`」という合成ホスト名で、これがコミットの作者情報・
コミットメッセージ・そして**現在のツリーのファイルパス自体**の3か所に
埋め込まれている。

ファイルパスへの埋め込みは、仮に消したい場合は単純な「今後そういうファイル名を
使わない」では済まず、既存のコミット履歴の書き換え(パス名の変更)が必要になる
(§7で手順の案のみ示す)。実害は乏しい(本人のPCの機種名が分かる程度)ため、
残すかどうかは安念の判断に委ねたい。

---

## 4. 大きいファイル

**結論: 50MBを超えるファイルは、全履歴のどのブランチにも一つもない。**

`git rev-list --objects --all`で全blobを列挙し、`git cat-file --batch-check`で
サイズを確認した。10MBを超えるものだけを挙げる(これ以上の大きさのファイルは存在しない):

| サイズ | パス | コミット | 現在も生きているブランチの先端にあるか |
|---|---|---|---|
| 31.6MB | `scripts/output/figures/local_placement_animation.mp4` | f63ea84d | **はい** — `claude/k-model-markov-probe`・`claude/inspiring-allen-ja4ele`・`claude/lindgren-population-evolution-08fat1`・`claude/markov-production-analysis`の現在の先端 |
| 22.6MB | `scripts/output/scent_spatial_sweep_n1000/scent_low/doeRyanoMacBook-Neo.local/seed_28_checkpoint.pkl` | a0bf7ac3 | `checkpoint-scratch-scent-spatial-n1000`のみ(削除候補、§5) |
| 14.7MB | 同ディレクトリの`seed_29_checkpoint.pkl` | a0bf7ac3 | 同上 |
| 11.7MB | `scripts/output/coupling_markov_production/viewer/k_model_edges.json` | 60a2ba8c | **はい** — `claude/k-model-markov-probe`の現在の先端 |

残り(5.2MB〜1.1MB程度)はすべて図(GIF/PNG)・pickleチェックポイント・`.npz`・
`.csv.gz`といったシミュレーション関連の成果物で、人のデータは含まれない。

`.git`ディレクトリ全体は705MB。これは一つの巨大ファイルによるものではなく、
1781個のコミットに分散した多数の小さなシミュレーション出力(CSV・pickle)の
積み重ねによるもの。`main`ブランチ自身の現在のツリーには10MB超のファイルは
一つもない——サイズは研究用の長期ブランチ・scratchブランチ側に集中している。

---

## 5. ブランチの整理案

全ブランチの一覧(最終コミット日、コミット総数、要約)。

| ブランチ | 最終コミット | 総コミット数 | 内容 | 案 |
|---|---|---|---|---|
| `main` | 2026-08-28 | 14 | 最初期の試作(2エージェントPD対戦、Lindgren記憶ゲノムの初期実装)。**論文の実データ・分析は含まない** | **残す**(ただし§「mainについて」参照——内容の刷新が必要) |
| `claude/markov-production-analysis` | 2026-10-23 | 1668 | Lindgren純粋戦略のMarkov構造・本番分析一式。引用コミット`ccc0b1b`・`3fe01f7`・`6f4ab3dc`・`f67c207`を含む | **残す** |
| `claude/k-model-markov-probe` | 2026-11-02 | 870 | Kモデル(二成分K)のMarkov構造・確証的分析一式。引用コミット`82203334`・`955a1c9`・`092ff792`・`a4457747`を含む | **残す** |
| `claude/lindgren-population-evolution-08fat1` | 2026-10-20(originの版) | 1615 | `claude/markov-production-analysis`の祖先(内容は完全にそちらに含まれる、独自コミットなし) | 削除可 |
| `claude/inspiring-allen-ja4ele` | 2026-10-14 | 760 | 空間N=1000モデルの初期実装。`markov-production-analysis`・`k-model-markov-probe`両方の祖先(両方に完全に含まれる) | 削除可 |
| `claude/evolutionary-personality-abm-tyh46w` | 2026-08-25 | 3 | 最初期の2エージェント対戦ループ。`main`自身の祖先(mainの#1マージで統合済み) | 削除可 |
| `claude/lindgren-memory-genome-afuv7x` | 2026-08-26 | 6 | Lindgren記憶ゲノムの初期実装。`main`自身の祖先(#2マージで統合済み) | 削除可 |
| `claude/memory-depth-duplication-uhwz4c` | 2026-08-27 | 9 | 記憶深度の重複変異。`main`自身の祖先(#3・#4マージで統合済み) | 削除可 |
| `checkpoint-scratch` | 2026-10-04 | 1(孤立コミット) | k=99スイープの再開用チェックポイント(.pklのみ、研究内容なし) | 削除可 |
| `checkpoint-scratch-scent-spatial` | 2026-10-11 | 1(孤立コミット) | 空間モデルの再開用チェックポイント(.pklのみ) | 削除可 |
| `checkpoint-scratch-scent-spatial-n1000` | 2026-10-14 | 1(孤立コミット) | N=1000空間モデルの再開用チェックポイント(.pklのみ)。**コミットメッセージに研究者のMacのホスト名を含む** | 削除可(ホスト名漏えいの観点からもなおさら) |

**削除可としたブランチを全部消しても、論文が引用する8コミットは全て
`claude/markov-production-analysis`と`claude/k-model-markov-probe`に残る**
(個別に`git branch -a --contains <hash>`で確認済み)。

補足: `claude/lindgren-population-evolution-08fat1`という名前は、このClaude
Codeセッションの設定(このリポジトリでの既定の作業ブランチ名)にも使われている。
リポジトリから削除する場合、そのセッション設定側の見直しも合わせて必要になる
(リポジトリの内容とは別の話)。

### `main`について(最重要・要判断)

`main`は2026-08-28で止まっており、論文が報告するMarkov構造分析・Kモデルの
確証的分析は一切含まれていない。このままGitHubを公開すると、訪問者が最初に
目にする既定ブランチの中身が、論文の内容と一致しない。3つの案を提示する
(いずれも実行していない):

- **(A)** `claude/markov-production-analysis`・`claude/k-model-markov-probe`の
  内容を`main`にマージ/反映し、`main`を実際の研究成果物にする。
- **(B)** `main`はそのまま(履歴の記録として)残しつつ、GitHub上のデフォルト
  ブランチの設定だけを`claude/markov-production-analysis`または
  `claude/k-model-markov-probe`に切り替える。
- **(C)** 両方の研究ライン(LindgrenのMarkov分析とKモデル)から最終版を
  整理・統合した新しい公開用ブランチ(例:`release/v1`)を作る。

どれを選ぶかは安念の判断による。

---

## 6. 公開用の下書き(`claude/public-release-prep`ブランチ、mainには未反映)

`claude/k-model-markov-probe`の現在の先端から分岐した新ブランチ
`claude/public-release-prep`に、以下の下書きを作成した(mainにはまだ入れていない):

- `README.md`: 研究内容の説明、人のデータを含まないことの明記、論文・事前登録
  (Zenodo DOIは`<!-- TODO -->`で空欄)への案内、再現手順(Python 3.11系、venv、
  `pip install -r requirements.txt -r requirements-dev.txt`、`pytest`、主要
  スクリプトの実行順序)。
- `LICENSE`: コード用のMITライセンス案(著作権者: Annen Yasumasa, 2026)。
- `LICENSE-DOCS`: `docs/`以下の文書・集計データ・図用のCC BY 4.0の案。
- `CITATION.cff`・`.zenodo.json`: 著者をAnnen, Yasumasa(Sapporo University of
  Health Sciences)として記入。バージョン・日付・DOIなど、まだ決まっていない
  項目は`TODO`で明示してある。

このブランチを`claude/k-model-markov-probe`から分岐させたのは、現時点で
README・CITATION.cffが参照しているファイル(`docs/PREREGISTRATION_k_model_markov.md`・
`docs/K_MODEL_MARKOV_MAC_RUNBOOK.md`・`docs/human_network_weights.csv`など)が
実際に存在するブランチだから。§5の「`main`について」の判断が決まった後、
最終的にどのブランチを公開既定ブランチにするかに合わせて、これらの下書きを
そちらへ移植する必要がある。

---

## 7. 履歴の書き換えが必要になった場合の手順案(未実行)

§3-2のホスト名漏えいを消したい場合の手順案。**まだ実行していない。**

1. **書き換え前に、現状をそのまま凍結して保存する**:
   - Zenodoへのアーカイブ(または少なくとも`git tag`)を、書き換え対象の
     各ブランチの**現在の**コミットに対して作る。これをしないと、論文の
     事前登録文書(`docs/PREREGISTRATION_k_model_markov.md`等)が記録している
     8つの確定コミットハッシュ(`ccc0b1b`・`3fe01f7`・`82203334`・`955a1c9`・
     `6f4ab3dc`・`f67c207`・`092ff792`・`a4457747`)が、書き換え後に**全て
     変わってしまい**、再現性の参照先を失う。
2. `git filter-repo`(`filter-branch`ではなく)を使い、1回のローカルの
   クローン上で、公開対象の全ブランチ(`main`・`claude/markov-production-analysis`・
   `claude/k-model-markov-probe`、§5の判断後に残すもの)に対して同時に:
   - `--path-rename`で、`doeRyanoMacBook-Neo.local`を含む約495個のパスを
     中立的な名前(例:`local-run`)に変更する。
   - コミットメッセージ中の191件のホスト名を、`--replace-message`等で
     中立的な文言に置き換える。
   - ファイル内容(JSON中の`"machine": "doeRyanoMacBook-Neo.local"`等、
     約495箇所)も`--replace-text`で置き換える。
3. この操作で影響する全コミットのハッシュが変わる。
   `docs/PREREGISTRATION_k_model_markov.md`・`docs/K_MODEL_MARKOV_DESIGN.md`
   などが記録している確定コミットハッシュを、新しいハッシュに更新するか
   (もしくは、これらの文書に「GitHub公開版ではhistory rewriteによりハッシュが
   変わっています。元のハッシュでの再現性はZenodoアーカイブ[DOI]を参照して
   ください」という注記を加える)。
4. `git push --force`で書き換え後の履歴を反映する(既存のフォーク・クローン・
   PRは無効になる。公開に切り替える直前に一度だけ行う)。
5. 書き換え後、本点検(§1〜4のgitleaks・人データ・大きいファイルのスキャン)を
   再実行し、ホスト名が実際に消えたこと・他に壊れていないことを確認してから
   公開する。

**所見**: ホスト名の漏えいは、パスワードや個人の被験者データのような重大な
機密ではなく、「研究者個人のPCの機種名が分かる」程度の軽微な漏えいである。
履歴の書き換えは上記の通りコミットハッシュの変更という実質的なコストを伴うため、
「そのまま公開して気にしない」という選択肢も合理的だと考える。最終的な判断は
安念に委ねる。

---

## 8. その他、判断が必要な点(公開してよい/いけないに関わる事実ではないが)

- `HBES2026_ABM骨子_二成分K.md`(`main`にのみ存在)は、学会発表に向けた構想・
  モデル設計の経緯をまとめた研究ノートで、個人情報・秘密情報は含まない。
  ただし「査読対策メモ」のような、著者自身の戦略的な覚書きも含まれている。
  内容自体に問題はないが、このまま公開するか、覚書き部分を外すかは
  安念の判断による。
