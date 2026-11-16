# 公開前点検(dpforest610/EvoPersonality_ABM)

実施日: 2026-11-16。対象: `git log --all`(全ブランチ・全リモート追跡ブランチ)の
全履歴。本リポジトリをGitHubで公開した場合、全ブランチの全コミット履歴が
閲覧可能になることを踏まえ、現在のデフォルトブランチ(`main`)だけでなく
全履歴を対象に調べた。この文書は点検結果の報告として始まり、安念の判断
(§9以降に記録)に沿って実行した作業も追記している。**ただし、GitHub上での
公開(Public化)・履歴の書き換え(`git filter-repo`等によるコミットハッシュの
変更)・ファイルの削除は、この文書を通じて一度も実行していない。**

## 結論(先に短く)

- **人のデータ・秘密情報・50MB超のファイル: いずれもゼロ。この3点は公開して問題ない。**
- **ホスト名(`doeRyanoMacBook-Neo.local`)について**: 安念の判断により、**残す。履歴の
  書き換えはしない**(§7は手順の案のみで、実行していない)。引用している8コミットの
  ハッシュを変えないことを最優先した。
- **`main`の内容について**: 安念の判断(案A)により、`claude/markov-production-analysis`と
  `claude/public-release-prep`(`claude/k-model-markov-probe`+公開用の下書き)を
  `--no-ff`でmainにマージ**済み**(force pushなし)。§9に作業記録。
- **不要なブランチ8本の削除について**: 安念がタグ2本のGitHub上への反映を確認した後、
  削除を試みたが、**このセッションの権限では削除がHTTP 403で拒否され、1本も削除
  できていない**(タグのpushと同じ制限)。削除前後の到達可能性の確認結果と、
  安念のMac/GitHub上で削除するための具体的な手順は§11。
- 公開(Public化)とZenodo連携の手順案は§10。

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

---

## 9. mainへのマージ作業の記録(安念の判断、案Aを実行)

安念の判断に基づき、以下を実行した(force pushはしていない)。

### 9-1. スナップショットタグ

マージ前の各ブランチの先端に注釈付きタグを作成した:

| タグ | 対象コミット |
|---|---|
| `snapshot/markov-production-analysis-20261116` | `6c43beb24762f82e44b9c147c50569d89270f2c7`(`claude/markov-production-analysis`の先端) |
| `snapshot/k-model-markov-probe-20261116` | `48536964d8a5431a630ab8f9fcc2be256cbb569c`(`claude/k-model-markov-probe`の先端) |

**このセッションの権限ではタグのpushがHTTP 403で拒否された**(ブランチのpushは
問題なくできる)。安念の指示により、タグは安念のMacからpushしてもらう形にして
マージ作業を先に進めた。ブランチ削除(§5)は、このタグがGitHub上に実際に
載ったことを確認してから実行する。

タグをpushするコマンド(安念のMacで、このリポジトリのクローン内で):

```bash
git fetch origin
git tag -a snapshot/markov-production-analysis-20261116 6c43beb24762f82e44b9c147c50569d89270f2c7 -m "Snapshot before merging into main for public release (2026-11-16)"
git tag -a snapshot/k-model-markov-probe-20261116 48536964d8a5431a630ab8f9fcc2be256cbb569c -m "Snapshot before merging into main for public release (2026-11-16)"
git push origin snapshot/markov-production-analysis-20261116 snapshot/k-model-markov-probe-20261116
```

(ローカルに既にこれらのコミットがなければ、先に`git fetch origin
claude/markov-production-analysis claude/k-model-markov-probe`で取得してから。)

### 9-2. マージの実行

`main`(`635ab597`)に対して、以下の順で`--no-ff`マージを実行した:

1. `git merge --no-ff origin/claude/markov-production-analysis` —
   **衝突なし**で完了。
2. `git merge --no-ff origin/claude/public-release-prep`
   (`claude/k-model-markov-probe`の全内容 + §6の公開用下書き一式を含む) —
   **2箇所で衝突**。

### 9-3. 衝突の解消

| ファイル | 衝突の種類 | 解消内容 |
|---|---|---|
| `.gitignore` | テキストの衝突(両側とも末尾に別々のパターンを追加しただけで、内容的な対立ではない) | **両方の内容を残した**。`claude/markov-production-analysis`側が追加した5行(`*_probe_age_*_census.csv.gz`等)と、`claude/public-release-prep`側が追加した6行(`coupling_markov_*/*_census.csv.gz`等)を、順にそのまま連結した。 |
| `evopersonality_abm/lindgren.py` | **サイレントな重複**(gitは衝突として検出しなかったが、`_table_ignores_oldest_round`と`effective_memory_depth`の2関数が、両ブランチが分岐後に独立に同じ内容を追加していたため、マージ後にファイル内に2回定義されていた) | 気づいて手動で修正。2つの定義は一字一句同じロジック(docstringの文言のみ、「Ported from...」と「NEW (not ported)」で異なる)だったため、後者(オリジナル側、`claude/markov-production-analysis`で最初に追加されたもの)を残し、前者(`claude/k-model-markov-probe`側の移植コピー)を削除して重複を解消した。関数の中身・動作は変えていない。 |

`docs/PREREGISTRATION_*`・`docs/DEVIATION_*`の文書は、この2つのマージのどちらでも
**衝突しなかった**(両ブランチに同名ファイルの重複がなかったため)。したがって
「衝突したらそのブランチの版をそのまま採用する」というルールを適用する場面
自体がなく、両文書とも一字も変更していない。

マージコミット後、全repoを対象に`grep`で関数・クラス定義の重複を再チェックし、
上記以外に重複が無いことを確認した。

### 9-4. README.mdのZenodo欄

安念の指示により、`README.md`の該当行を次のように書き換えた
(番号はまだ不明なのでTODOのまま):

```
- データ・コードのアーカイブ(Zenodo DOI): 10.5281/zenodo.TODO <!-- TODO: 登録後に正式な番号に差し替え -->
```

### 9-5. 引用コミット8つの祖先確認

`git merge-base --is-ancestor <hash> main`で、マージ後のmainに対して全て確認した:

| コミット | 結果 |
|---|---|
| `ccc0b1b` | ○ mainの祖先 |
| `3fe01f7` | ○ mainの祖先 |
| `82203334` | ○ mainの祖先 |
| `955a1c9` | ○ mainの祖先 |
| `6f4ab3dc` | ○ mainの祖先 |
| `f67c207` | ○ mainの祖先 |
| `092ff792` | ○ mainの祖先 |
| `a4457747` | ○ mainの祖先 |

**8つ全て確認済み。**

### 9-6. テスト結果

```
554 passed in 23.08s
```

マージ後のmainで全テストが通過した。

### 9-7. push

上記9-1〜9-6が全て完了した後に、`main`をpushする(force pushではなく、
通常のfast-forward push——`origin/main`からの分岐がないため)。

---

## 10. 公開(Public化)・GitHub Release・Zenodo連携の手順(安念が手で実行する)

以下は手順の案のみ。実行は安念が行う。

1. **Zenodo側でGitHub連携を先にONにする**(これが一番重要:GitHub Releaseを
   作ってからZenodo連携をONにしても、その回のReleaseはアーカイブされない):
   - https://zenodo.org にGitHubアカウントでログイン。
   - https://zenodo.org/account/settings/github/ を開き、
     `dpforest610/EvoPersonality_ABM`を探してトグルをONにする
     (リポジトリをPublicにした後でないと一覧に出ない場合がある——その場合は
     先に手順2でPublicにしてから戻ってくる)。
2. **リポジトリをPublicにする**: GitHubの当該リポジトリ →
   Settings → General → 一番下のDanger Zone →
   "Change repository visibility" → Public。
3. **バージョンタグを作る**(例:`v1.0.0`、`main`の最新コミットに対して)。
4. **GitHub Releaseを作る**: Releases → "Draft a new release" →
   手順3のタグを選択 → 本文に論文・プレプリントへの案内を書いて公開。
   これでZenodoがWebhookを受け取り、自動でスナップショットを作成しDOIを発行する
   (このリポジトリ直下の`.zenodo.json`のメタデータが使われる)。
5. 発行されたDOIを確認し、`README.md`の
   `10.5281/zenodo.TODO`と`CITATION.cff`の`doi`欄(コメントアウトを外す)を
   実際の番号に書き換えてpushする。
6. `CITATION.cff`の`version`・`date-released`・`preferred-citation`欄
   (論文の出版先・DOIが決まったもの)も、判明した時点で更新する。

---

## 11. 不要なブランチ8本の削除(安念の確認後に着手、ただし削除は未完了)

### 11-1. タグのGitHub反映確認

安念がMacから2本のタグをpushし、GitHub上で以下の通り確認された:

| タグ | 指しているコミット |
|---|---|
| `snapshot/markov-production-analysis-20261116` | `6c43beb24762f82e44b9c147c50569d89270f2c7` |
| `snapshot/k-model-markov-probe-20261116` | `48536964d8a5431a630ab8f9fcc2be256cbb569c` |

このセッション側でも`git fetch origin --tags`で同じ内容を確認した
(軽量タグ。§9で作成した注釈付きタグとはタグオブジェクト自体は別物になるが、
指しているコミットが一致していれば十分なので、作り直していない)。

### 11-2. 削除前の到達可能性の確認

引用している8コミットそれぞれについて、`origin/main`と2本のタグそれぞれから
`git merge-base --is-ancestor`で確認した:

| コミット | `origin/main`の祖先か | `snapshot/markov-production-analysis-20261116`の祖先か | `snapshot/k-model-markov-probe-20261116`の祖先か |
|---|---|---|---|
| `ccc0b1b` | ○ | ○ | - |
| `3fe01f7` | ○ | ○ | - |
| `82203334` | ○ | - | ○ |
| `955a1c9` | ○ | - | ○ |
| `6f4ab3dc` | ○ | ○ | - |
| `f67c207` | ○ | ○ | - |
| `092ff792` | ○ | - | ○ |
| `a4457747` | ○ | - | ○ |

**8つ全てが`origin/main`の祖先であり、かつ少なくとも一方のタグの祖先でもある**
(main・タグの二重の経路で到達可能)。あわせて、2本のタグが指すコミット自体も
`origin/main`の祖先であることを確認した。削除対象の8ブランチを消しても、
mainとタグのどちらか一方さえ残っていれば、この8コミット全てに到達できる。

### 11-3. 削除の実行 ―― このセッションの権限ではできなかった

安念の指示通り、まず以下の7本から削除を試みた(`claude/lindgren-population-evolution-08fat1`
は指示通り最後に回す予定だった):

- `claude/inspiring-allen-ja4ele`
- `claude/evolutionary-personality-abm-tyh46w`
- `claude/lindgren-memory-genome-afuv7x`
- `claude/memory-depth-duplication-uhwz4c`
- `checkpoint-scratch`
- `checkpoint-scratch-scent-spatial`
- `checkpoint-scratch-scent-spatial-n1000`

`git push origin --delete <branch>`をそれぞれに対して実行したが、**7本とも
HTTP 403で拒否された**(タグのpushを拒否されたのと同じ、このセッションの
git認証の権限の問題)。GitHub用のMCPツールにもブランチ削除に相当するツールは
なかった。README(`/root/.ccr/README.md`)の指示(「403/407はリトライしたり
回避したりせず、報告する」)に従い、再試行や別経路での削除は試みていない。

削除を試みた後、`git ls-remote --heads origin`で7本とも**無傷でそのまま
残っている**ことを確認した。**この時点で、8本のブランチは1本も削除されて
いない。**

### 11-4. 安念のMac/GitHub側で削除する手順

以下を安念のMacから(このリポジトリのクローン内で)実行するか、GitHubの
Webで該当ブランチのゴミ箱アイコンから削除してください。**指示通り、
`claude/lindgren-population-evolution-08fat1`は最後に。**

```bash
git push origin --delete claude/inspiring-allen-ja4ele
git push origin --delete claude/evolutionary-personality-abm-tyh46w
git push origin --delete claude/lindgren-memory-genome-afuv7x
git push origin --delete claude/memory-depth-duplication-uhwz4c
git push origin --delete checkpoint-scratch
git push origin --delete checkpoint-scratch-scent-spatial
git push origin --delete checkpoint-scratch-scent-spatial-n1000
# ここまで確認してから最後に:
git push origin --delete claude/lindgren-population-evolution-08fat1
```

削除後、§11-2の表と同じ確認(8コミット全てが`main`・タグいずれかの祖先であること)
を再度行うことを勧める——このセッションでも、削除完了の報告を受けてから
改めて`git fetch --prune`の上で確認し、この文書に追記する。

### 11-5. 補足

`claude/lindgren-population-evolution-08fat1`というブランチ名は、このClaude
Codeセッションのこのリポジトリでの既定の作業ブランチ名としても設定されている
(§5の補足で既出)。削除後、以後のセッションでこの名前が既定として指定され
続ける場合、新規作成扱いになる(内容が空の状態から始まる)点に留意してください。
