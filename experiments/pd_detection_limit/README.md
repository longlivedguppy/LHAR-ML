# 正解付き疑似PDデータによる検出限界評価

この実験は、真のPD50が既知の1Dプロファイルを使って、LHAR-MLの
**algorithmic detection limit（アルゴリズム検出限界）**を評価する。
実測画像を処理する `main.py` の入出力ディレクトリとは分離されており、
実際の膜厚やcycle数に対する **optical measurement detection limit** を
直接主張するものではない。

## 現在の範囲

Phase 1を実装している。

```text
constant baseline + sigmoid PD signal + Gaussian random noise
```

同じnoisy inputを次の3経路へ流し、共通のproduction候補検出処理で評価する。

1. Raw
2. 既存Median+Gaussian（kernel=5、sigma=5）
3. 既存Huber MRF（既定lambda=1）

`lambda=1` は `compare_mrf_profile.py` の既存予備比較で使われていた既定値を
引き継いだものであり、最適化済みの値ではない。`mrf_regularization` で変更できる。

production候補検出は、`np.gradient`、絶対値、設定可能な先頭領域の除外、
`scipy.signal.find_peaks(height=0, distance=15)`、高さ上位2候補の順で処理する。
候補がない場合に `argmax` を返す既存挙動も変えていない。
理論プロファイルを全域評価するPhase 1では `start_distance=0` とし、除外領域を
図示しない。実測用 `main.py` の既定50 pxは変更していない。

## 推奨する実行順

リポジトリルートから、まず1条件を確認する。

```powershell
python -m experiments.pd_detection_limit.run_benchmark --mode single
```

出力は条件を含む決定的なtrialディレクトリへ整理する。

```text
outputs/phase1/single/
└── trial_seed_<seed>_amp_<amplitude>_width_<width>_noise_<std>_mrf_lambda_<lambda>/
    ├── data/
    │   ├── profiles.csv
    │   ├── trial_method_results.csv
    │   └── method_summary.csv
    ├── figures/
    │   ├── 01_raw_detection.png
    │   ├── 02_median_gaussian_detection.png
    │   └── 03_huber_mrf_detection.png
    └── metadata/
        └── run_metadata.json
```

各図は1手法だけを表示する。上段はnoise-free profile、全手法共通のnoisy input、
該当するdenoise結果、真のPD50、PD prediction、下段は該当手法の絶対微分と
PD predictionである。`PD prediction` はdetectorが算出したPD位置を意味する。

次に少数trialでparameter sweepとMonte Carlo経路を確認する。

```powershell
python -m experiments.pd_detection_limit.run_benchmark --mode sweep --n-trials 5
```

確認後、設定どおりの100 trialを実行する。

```powershell
python -m experiments.pd_detection_limit.run_benchmark --mode sweep
```

`--no-plots` でCSV/JSONだけを生成できる。`--output-dir` で出力先を分離できる。
Sweepも `run_seed_<seed>_trials_<n>_mrf_lambda_<lambda>/` の下を `data/`、`figures/heatmaps/`、
`figures/example_signal_trial/`、`metadata/` に分ける。

## 設定

`configs/phase1.json` を標準ライブラリだけで読み込む。PyYAMLは追加していない。
主な設定は次のとおり。

- `profile_length=1201`: 現行ROI（y=1400..200、両端を含む）と同じ長さ
- `pd50_true=600`: 端部フィルタ影響を避ける中央位置
- `baseline=30000`: 16-bit実測Gray Valueの代表的なオーダー
- `amplitudes=0..10000`: blankから明瞭な境界までを含む広い検証範囲
- `widths=3, 8, 20`: 急峻、中間、緩やかな境界
- `noise_stds=150, 300, 600`: 実測プロファイル数例の隣接差MAD由来尺度
  （約150～300）を中心に、より厳しい条件も含める
- `tolerance_px=5`: 物理的に確定した値ではなく、必ず変更可能な評価設定
- `n_trials=100`: 本評価用。デバッグ時はCLIで5などに上書きする
- `start_distance=0`: 理論Phase 1では全プロファイルを評価する

確認した実測処理済みCSVでは、ROI長は762または1201 px、Gray Valueは概ね
数万、プロファイル両端のレベル差は約1.5k～10kだった。これらは初期範囲の
オーダーを決める参考値であり、光学的な校正値ではない。

同じ `trial_index` では全parameter条件に同じseedを使う。各条件間で同じ
Gaussian noise realizationを共有するためであり、全手法にも全く同じnoisy
inputが渡る。乱数は `np.random.default_rng(seed)` と `Generator.normal()` を使う。

## trial結果と集約指標

各trialは、phase、seed、真値、amplitude、width、baseline、背景傾斜、noise、
手法、filter/MRF設定、推定PD、signed/absolute error、最大絶対微分、上位2候補、
`argmax` fallback使用有無を保存する。

1 trialから `raw`、`median_gaussian`、`huber_mrf` の3行を
`trial_method_results.csv` に保存する。singleでは各行に対応する3つの図を保存する。
Sweepでは全trialの行を直接すべて作図せず、最初の信号ありtrialだけを3図で診断し、
全行を条件・手法単位に集約してヒートマップへ変換する。

信号ありでは次を集約する。

- MAE
- signed bias
- PD prediction（算出されたPD位置）の標準偏差
- `abs(estimated - true) <= tolerance_px` のcorrect detection rate

`amplitude=0` は真のPDが存在しないblankである。設定上の `pd50_true` は
再現性のため記録するが、blankのsigned/absolute errorとcorrect detection rateは
定義しない。現行detectorが返した位置は `current detector output` として残し、
候補が返ればfalse positiveとして数える。現行fallback構造ではfalse positive
rateが1になり得ること自体が、将来のBelow Detection Limit判定設計に必要な結果である。

## Phase 3/4への境界

`data/extracted_noise/` は、将来実測したbackground/artifact profileを置く場所で
ある。現在は抽出データを仮定せず、ローダーのインターフェースだけを用意した。
人工的な「実測ノイズ」は生成しない。0 cycleデータも純粋なnoiseとは断定せず、
構造、光学系、照明、固定パターン、センサーを含み得るmeasured
background/artifactとして扱う。
