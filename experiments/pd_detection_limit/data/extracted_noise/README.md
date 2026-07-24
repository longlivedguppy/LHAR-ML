# Measured background / artifact profiles

Phase 3で実測画像を確認した後に、由来と抽出手順を記録したプロファイルを置く。
0 cycleを純粋なnoiseとは見なさない。現時点では実測プロファイルは未登録であり、
このディレクトリの内容を人工生成して代用しない。

将来のローダーは、1Dの `.npy` またはヘッダーなし1列 `.csv` を受け付ける。
各データには元画像、ROI、line aggregation、detrendingの有無を別途記録すること。
