"""第二章 精密kスイープ 右側延長: k=99(全結合と数学的に同一構造だが同一パイプライン),
64, 80 を追加。

既存の run_opponent_count_precision_sweep.py と完全に同一のパイプライン
(spatial.py の1D環状, run_coupled_evolution_opponent_count, n_phases=1,
移動なし, 毎世代シャッフル, グローバル適応度比例繁殖, population=100,
その他パラメータ全て既存スイープと同一)で、出力先も同じ
scripts/output/opponent_count_precision_sweep/ 配下(k99/, k64/, k80/)
に書き込むため、後段の解析は既存9点+この3点をあわせて12点として
一元的に扱える。

事前検証(N=100, k=99): 全100個体が自分以外の全99個体とちょうど1回ずつ
重複なく接続され(C(100,2)=4950エッジ、全degree=99、自己ループなし)、
これはチャット上で実行・確認済み。

実行優先順位: k=99(判定に決定的)→ k=64 → k=80。
各 n=50シード × 5000世代、gen>=2000で評価(既存と同一)。

チェックポイント・再開・git push機構は既存スイープと完全に同一。

Usage:
    python3 scripts/run_opponent_count_precision_sweep_extension.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.run_opponent_count_precision_sweep as base

base.K_VALUES = [99, 64, 80]  # priority order: k=99 is decisive, run first

if __name__ == "__main__":
    base.main()
