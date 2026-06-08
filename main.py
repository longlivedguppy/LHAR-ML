import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# srcフォルダ内の自作モジュールから関数をインポート
from src.data_loader import extract_multi_line_profile
from src.denoise import apply_smoothing_filter, apply_median_filter, apply_median_gaussian_filter
# 2D行列のフィルタリングにScipy関数を直接使用
from scipy.signal import medfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

def main():
    # --- Interactive Folder Selection ---
    print("処理モードを選択してください [1: 生画像から処理 (raw), 2: 既存のクリーンデータを使用 (processed), 3: サマリーのみ生成 (summary only)]: ")
    mode = input("> ")

    if mode == '1':
        scan_dir = "data/raw"
    elif mode in ('2', '3'):
        scan_dir = "data/processed"
    else:
        print("エラー: 無効なモードが選択されました。プログラムを終了します。")
        return

    if not os.path.isdir(scan_dir):
        print(f"エラー: スキャン対象のディレクトリが見つかりません: '{scan_dir}'")
        return

    try:
        available_dirs = sorted([d for d in os.listdir(scan_dir) if os.path.isdir(os.path.join(scan_dir, d))])
    except Exception as e:
        print(f"ディレクトリのスキャン中にエラーが発生しました: {e}")
        return

    if not available_dirs:
        print(f"エラー: '{scan_dir}' 内に処理対象のフォルダが見つかりません。")
        return

    print("\n解析したいフォルダの番号を入力してください:")
    for i, dir_name in enumerate(available_dirs):
        print(f"[{i}] {dir_name}")

    try:
        selected_index = int(input("> "))
        if not 0 <= selected_index < len(available_dirs):
            raise IndexError
        target_dataset_dir = available_dirs[selected_index]
        print(f"-> '{target_dataset_dir}' が選択されました。\n")
    except (ValueError, IndexError):
        print("エラー: 無効な番号が入力されました。プログラムを終了します。")
        return

    # ==========================================
    # 解析・MLの実行コントロールスイッチ (True/False)
    # ==========================================
    RUN_DATA_EXTRACTION = (mode == '1')   # モード1の場合のみ生画像からデータを抽出
    ONLY_SUMMARY        = (mode == '3')   # サマリー画像のみを出力する
    RUN_MEDIAN_FILTER   = False   # メディアンフィルタによるスパイクノイズ除去を行う
    RUN_GAUSSIAN_FILTER = False   # ガウシアンフィルタによる平滑化を行う
    RUN_MEDIAN_GAUSSIAN_FILTER = False # メディアンフィルタ適用後にガウシアンフィルタをかける
    RUN_DERIVATIVE_ANALYSIS = True   # 微分解析とグラフ出力を行う
    RUN_ML_REGRESSION   = False  # 【今後実装】機械学習による膜厚・深さ予測
    RUN_EDGE_DETECTION  = False  # 【今後実装】製膜境界（端）の自動検出

    # ==========================================
    # 処理パラメータとパスの設定
    # ==========================================
    N_LINES = 5
    LINE_GAP = 4
    START_DISTANCE = 50  # 微分解析の開始位置（ピクセル数）
    TOLERANCE_PX = 40.0  # 全体中央値からの許容ズレ幅（ピクセル）

    # 入力パス設定
    raw_dir = os.path.join("data", "raw", target_dataset_dir)
    
    # 出力パスの動的生成
    base_processed_dir = os.path.join("data", "processed", target_dataset_dir, f"{N_LINES}lines")
    base_plot_dir = os.path.join("output_plots", target_dataset_dir, f"{N_LINES}lines")

    path_structure = {
        "1d_average": {
            "data": os.path.join(base_processed_dir, "1d_average"),
            "plots": os.path.join(base_plot_dir, "1d_average")
        },
        "2d_surface": {
            "plots": os.path.join(base_plot_dir, "2d_surface")
        }
    }
    
    # ImageJの長方形選択（Width: 7, Height: 761, X: 506, Y: 142）に基づく座標
    # x1がサンプリング開始X座標となる
    roi_coords = {"x1": 509, "y1": 142, "x2": 509, "y2": 903}
    
    # 必要な全フォルダを自動生成
    filter_types = ["median", "gaussian", "median_gaussian", "derivative", os.path.join("derivative", "raw"), os.path.join("derivative", "corrected")]
    for proc_type, paths in path_structure.items():
        for category, base_path in paths.items():
            for filter_type in filter_types:
                os.makedirs(os.path.join(base_path, filter_type), exist_ok=True)

    # ==========================================
    # 処理対象のファイルリストを取得
    # ==========================================
    target_files = [] # (img_path, base_filename)のリスト
    
    if mode == '1':
        img_files = glob.glob(os.path.join(raw_dir, "*.tiff")) + glob.glob(os.path.join(raw_dir, "*.tif"))
        for f in img_files:
            target_files.append((f, os.path.splitext(os.path.basename(f))[0]))
        if not target_files:
            print(f"エラー: {raw_dir} 内に処理対象の画像ファイル（.tiff, .tif）が見つかりません。")
            return
        print(f"--- {target_dataset_dir} フォルダ内の {len(target_files)} 個の画像を処理します ---")
        
    elif mode in ('2', '3'):
        # どのフィルタフラグもTrueでない場合、ユーザーに選択を促す
        if not any([RUN_MEDIAN_FILTER, RUN_GAUSSIAN_FILTER, RUN_MEDIAN_GAUSSIAN_FILTER]):
            print("\n読み込むクリーンデータのフィルタタイプを選択してください:")
            filter_options = {
                "1": ("median", "RUN_MEDIAN_FILTER"),
                "2": ("gaussian", "RUN_GAUSSIAN_FILTER"),
                "3": ("median_gaussian", "RUN_MEDIAN_GAUSSIAN_FILTER")
            }
            print("[1] median\n[2] gaussian\n[3] median_gaussian")
            
            try:
                choice = input("> ")
                if choice in filter_options:
                    filter_name, flag_name = filter_options[choice]
                    print(f"-> '{filter_name}' データを選択しました。\n")
                    # 選択に基づいてフラグを動的にTrueにする
                    if flag_name == "RUN_MEDIAN_FILTER": RUN_MEDIAN_FILTER = True
                    elif flag_name == "RUN_GAUSSIAN_FILTER": RUN_GAUSSIAN_FILTER = True
                    elif flag_name == "RUN_MEDIAN_GAUSSIAN_FILTER": RUN_MEDIAN_GAUSSIAN_FILTER = True
                else:
                    raise ValueError
            except (ValueError, KeyError):
                print("エラー: 無効な選択です。プログラムを終了します。")
                return

        search_dirs = []
        if RUN_MEDIAN_GAUSSIAN_FILTER: search_dirs.append((os.path.join(path_structure["1d_average"]["data"], "median_gaussian"), "median_gaussian_"))
        elif RUN_GAUSSIAN_FILTER: search_dirs.append((os.path.join(path_structure["1d_average"]["data"], "gaussian"), "gaussian_"))
        elif RUN_MEDIAN_FILTER: search_dirs.append((os.path.join(path_structure["1d_average"]["data"], "median"), "median_"))
        
        search_dir, prefix = search_dirs[0]
        csv_files = glob.glob(os.path.join(search_dir, "*.csv"))
        for f in csv_files:
            b_name = os.path.splitext(os.path.basename(f))[0]
            if b_name.startswith(prefix):
                b_name = b_name[len(prefix):]
            target_files.append((None, b_name))
            
        if not target_files:
             print(f"エラー: {search_dir} 内に処理対象のCSVファイルが見つかりません。事前にモード1で処理を行ってください。")
             return
        if ONLY_SUMMARY:
            print(f"--- {target_dataset_dir} フォルダ内の {len(target_files)} 個のCSVデータからサマリー画像を生成します ---")
        else:
            print(f"--- {target_dataset_dir} フォルダ内の {len(target_files)} 個のCSVデータを微分解析します ---")

    # ==========================================
    # 🏁 実際の解析処理の実行パート (ループ処理)
    # ==========================================
    progress_bar = tqdm(target_files, unit="file", desc="Processing")
    
    summary_data = {}  # 全角度サマリーヒートマップ用データ格納辞書

    for img_path, base_filename in progress_bar:
        progress_bar.set_postfix_str(f"{base_filename}", refresh=True)

        avg_intensities = None  # 各ファイル処理の前に初期化
        matrix_intensities = None
        
        median_intensities = None
        clean_intensities = None
        hybrid_intensities = None

        # --- モード1: 生画像からデータ抽出とフィルタリング ---
        if mode == '1':
            if RUN_DATA_EXTRACTION:
                try:
                    avg_intensities, matrix_intensities = extract_multi_line_profile(
                        img_path, roi_coords, num_lines=N_LINES, line_gap=LINE_GAP)
                except Exception as e:
                    tqdm.write(f"\nエラー: {e}")
                    continue
            
            if avg_intensities is None:
                tqdm.write(f"\nスキップ ({base_filename}): 輝度データがありません。")
                continue 

            # 2-A. メディアンフィルタ処理
            if RUN_MEDIAN_FILTER:
                csv_path_1d = os.path.join(path_structure["1d_average"]["data"], "median", f"median_{base_filename}.csv")
                plot_path_1d = os.path.join(path_structure["1d_average"]["plots"], "median", f"median_{base_filename}.png")
                median_intensities = apply_median_filter(avg_intensities, csv_path_1d, kernel_size=5)
                plt.figure(figsize=(10, 6))
                plt.plot(avg_intensities, color="gray", alpha=0.5, label="Raw Average")
                plt.plot(median_intensities, color="blue", linewidth=2, label="Median Filtered Data")
                plt.title(f"1D Average: Median Filter - {base_filename}")
                plt.xlabel("Distance (pixels)"); plt.ylabel("Gray Value"); plt.legend(); plt.grid(True)
                plt.savefig(plot_path_1d, dpi=300); plt.close()

                plot_path_3d = os.path.join(path_structure["2d_surface"]["plots"], "median", f"3d_surface_median_{base_filename}.png")
                filtered_matrix = np.apply_along_axis(medfilt, 0, matrix_intensities, kernel_size=5)
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                rows, cols = filtered_matrix.shape
                X, Y = np.meshgrid(np.arange(rows), np.arange(cols))
                Z = filtered_matrix.T
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
                fig.colorbar(surf, ax=ax, label='Gray Value', shrink=0.5, aspect=10)
                ax.set_title(f"3D Surface: Median Filter - {base_filename}")
                ax.set_xlabel("Distance (pixels)"); ax.set_ylabel("Line Number"); ax.set_zlabel("Gray Value")
                ax.view_init(elev=30, azim=-60)
                plt.savefig(plot_path_3d, dpi=300); plt.close()
            
            # 2-B. ガウシアンフィルタ処理
            if RUN_GAUSSIAN_FILTER:
                csv_path_1d = os.path.join(path_structure["1d_average"]["data"], "gaussian", f"gaussian_{base_filename}.csv")
                plot_path_1d = os.path.join(path_structure["1d_average"]["plots"], "gaussian", f"gaussian_{base_filename}.png")
                clean_intensities = apply_smoothing_filter(avg_intensities, csv_path_1d, method="gaussian")
                plt.figure(figsize=(10, 6))
                plt.plot(avg_intensities, color="gray", alpha=0.5, label="Raw Average")
                plt.plot(clean_intensities, color="black", linewidth=2, label="Gaussian Filtered Data")
                plt.title(f"1D Average: Gaussian Filter - {base_filename}")
                plt.xlabel("Distance (pixels)"); plt.ylabel("Gray Value"); plt.legend(); plt.grid(True)
                plt.savefig(plot_path_1d, dpi=300); plt.close()

                plot_path_3d = os.path.join(path_structure["2d_surface"]["plots"], "gaussian", f"3d_surface_gaussian_{base_filename}.png")
                filtered_matrix = np.apply_along_axis(gaussian_filter1d, 0, matrix_intensities, sigma=5)
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                rows, cols = filtered_matrix.shape
                X, Y = np.meshgrid(np.arange(rows), np.arange(cols))
                Z = filtered_matrix.T
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
                fig.colorbar(surf, ax=ax, label='Gray Value', shrink=0.5, aspect=10)
                ax.set_title(f"3D Surface: Gaussian Filter - {base_filename}")
                ax.set_xlabel("Distance (pixels)"); ax.set_ylabel("Line Number"); ax.set_zlabel("Gray Value")
                ax.view_init(elev=30, azim=-60)
                plt.savefig(plot_path_3d, dpi=300); plt.close()
                
            # 2-C. ハイブリッドフィルタ処理
            if RUN_MEDIAN_GAUSSIAN_FILTER:
                csv_path_1d = os.path.join(path_structure["1d_average"]["data"], "median_gaussian", f"median_gaussian_{base_filename}.csv")
                plot_path_1d = os.path.join(path_structure["1d_average"]["plots"], "median_gaussian", f"median_gaussian_{base_filename}.png")
                hybrid_intensities = apply_median_gaussian_filter(avg_intensities, csv_path_1d, kernel_size=5, sigma=5)
                plt.figure(figsize=(10, 6))
                plt.plot(avg_intensities, color="gray", alpha=0.5, label="Raw Average")
                plt.plot(hybrid_intensities, color="red", linewidth=2, label="Median + Gaussian Filtered Data")
                plt.title(f"1D Average: Hybrid Filter - {base_filename}")
                plt.xlabel("Distance (pixels)"); plt.ylabel("Gray Value"); plt.legend(); plt.grid(True)
                plt.savefig(plot_path_1d, dpi=300); plt.close()

                plot_path_3d = os.path.join(path_structure["2d_surface"]["plots"], "median_gaussian", f"3d_surface_median_gaussian_{base_filename}.png")
                median_filtered_matrix = np.apply_along_axis(medfilt, 0, matrix_intensities, kernel_size=5)
                hybrid_filtered_matrix = np.apply_along_axis(gaussian_filter1d, 0, median_filtered_matrix, sigma=5)
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                rows, cols = hybrid_filtered_matrix.shape
                X, Y = np.meshgrid(np.arange(rows), np.arange(cols))
                Z = hybrid_filtered_matrix.T
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
                fig.colorbar(surf, ax=ax, label='Gray Value', shrink=0.5, aspect=10)
                ax.set_title(f"3D Surface: Hybrid Filter - {base_filename}")
                ax.set_xlabel("Distance (pixels)"); ax.set_ylabel("Line Number"); ax.set_zlabel("Gray Value")
                ax.view_init(elev=30, azim=-60)
                plt.savefig(plot_path_3d, dpi=300); plt.close()

        # --- モード2, 3: 既存のフィルタ済みCSVからデータ読み込み ---
        elif mode in ('2', '3'):
            if RUN_MEDIAN_FILTER:
                csv_path = os.path.join(path_structure["1d_average"]["data"], "median", f"median_{base_filename}.csv")
                if os.path.exists(csv_path):
                    median_intensities = pd.read_csv(csv_path)["Median_Intensity"].values
            
            if RUN_GAUSSIAN_FILTER:
                csv_path = os.path.join(path_structure["1d_average"]["data"], "gaussian", f"gaussian_{base_filename}.csv")
                if os.path.exists(csv_path):
                    clean_intensities = pd.read_csv(csv_path)["Clean_Intensity"].values
                    
            if RUN_MEDIAN_GAUSSIAN_FILTER:
                csv_path = os.path.join(path_structure["1d_average"]["data"], "median_gaussian", f"median_gaussian_{base_filename}.csv")
                if os.path.exists(csv_path):
                    hybrid_intensities = pd.read_csv(csv_path)["Median_Gaussian_Intensity"].values

        # --- 3. 微分解析ステップ (RUN_DERIVATIVE_ANALYSIS) ---
        if RUN_DERIVATIVE_ANALYSIS:
            active_filters = []
            if RUN_MEDIAN_FILTER and median_intensities is not None:
                active_filters.append(("median", median_intensities, "Median Filter"))
            if RUN_GAUSSIAN_FILTER and clean_intensities is not None:
                active_filters.append(("gaussian", clean_intensities, "Gaussian Filter"))
            if RUN_MEDIAN_GAUSSIAN_FILTER and hybrid_intensities is not None:
                active_filters.append(("median_gaussian", hybrid_intensities, "Hybrid Filter"))
                
            for f_name, intensities, f_title in active_filters:
                deriv_csv_path = os.path.join(path_structure["1d_average"]["data"], "derivative", f"deriv_{f_name}_{base_filename}.csv")
                deriv_plot_path = os.path.join(path_structure["1d_average"]["plots"], "derivative", f"deriv_{f_name}_{base_filename}.png")
                
                derivative = np.gradient(intensities)
                abs_derivative = np.abs(derivative)
                distance = np.arange(len(intensities))
                
                # --- START_DISTANCE未満のデータをクリップ（ピーク探索から除外） ---
                abs_derivative_clipped = abs_derivative.copy()
                abs_derivative_clipped[:START_DISTANCE] = 0
                
                # scipy.signal.find_peaks を用いて独立した上位2つのピークを取得
                peaks, properties = find_peaks(abs_derivative_clipped, height=0, distance=15)
                if len(peaks) >= 2:
                    sorted_idx = np.argsort(properties['peak_heights'])[::-1]
                    p1_x = peaks[sorted_idx[0]]
                    p1_v = properties['peak_heights'][sorted_idx[0]]
                    p2_x = peaks[sorted_idx[1]]
                    p2_v = properties['peak_heights'][sorted_idx[1]]
                elif len(peaks) == 1:
                    p1_x = peaks[0]
                    p1_v = properties['peak_heights'][0]
                    p2_x = p1_x
                    p2_v = 0
                else:
                    p1_x = np.argmax(abs_derivative_clipped)
                    p1_v = abs_derivative[p1_x]
                    p2_x = p1_x
                    p2_v = 0
                
                # 個別プロット用は1stピークを使用
                best_peak_idx = p1_x
                best_peak_val = p1_v

                # --- サマリー用の2D配列とピーク位置をメモリに格納 ---
                if f_name not in summary_data:
                    summary_data[f_name] = {
                        "intensities": [], "p1_x": [], "p1_v": [], "p2_x": [], "p2_v": [],
                        "base_filenames": [], "distances": [], "abs_derivs": [], "raw_intensities": [], "f_titles": []
                    }
                
                # Min-Max正規化 (0.0〜1.0)
                int_min, int_max = np.min(intensities), np.max(intensities)
                if int_max > int_min:
                    normalized_intensity = (intensities - int_min) / (int_max - int_min)
                else:
                    normalized_intensity = np.zeros_like(intensities)  # ゼロ除算回避
                    
                summary_data[f_name]["intensities"].append(normalized_intensity)
                summary_data[f_name]["p1_x"].append(p1_x)
                summary_data[f_name]["p1_v"].append(p1_v)
                summary_data[f_name]["p2_x"].append(p2_x)
                summary_data[f_name]["p2_v"].append(p2_v)
                summary_data[f_name]["base_filenames"].append(base_filename)
                summary_data[f_name]["distances"].append(distance)
                summary_data[f_name]["abs_derivs"].append(abs_derivative)
                summary_data[f_name]["raw_intensities"].append(intensities)
                summary_data[f_name]["f_titles"].append(f_title)
                
                if not ONLY_SUMMARY:
                    df_deriv = pd.DataFrame({"Distance": distance, "Intensity": intensities, "Derivative": derivative})
                    df_deriv.to_csv(deriv_csv_path, index=False)

    # ==========================================
    # 全角度微分サマリーヒートマップの生成と保存
    # ==========================================
    if RUN_DERIVATIVE_ANALYSIS and summary_data:
        print("\n[サマリー生成] 全体統計ベースの一括検疫・補間とヒートマップ生成を実行中...")
        for f_name, data in summary_data.items():
            intensities_matrix = np.array(data["intensities"])  # Shape: [num_images, 761]
            p1_x = np.array(data["p1_x"], dtype=float)
            p1_v = np.array(data["p1_v"], dtype=float)
            p2_x = np.array(data["p2_x"], dtype=float)
            p2_v = np.array(data["p2_v"], dtype=float)
            y_indices = np.arange(len(p1_x))
            
            # --- 全体中央値の計算 (アンカー) ---
            median_1st_x = np.median(p1_x)
            dynamic_min_threshold = np.median(p1_v) * 0.5
            
            # --- 一括検疫 & 2次ピーク救済 ---
            final_peaks = np.full(len(p1_x), np.nan)
            status = np.zeros(len(p1_x), dtype=int)
            
            for i in range(len(p1_x)):
                if abs(p1_x[i] - median_1st_x) <= TOLERANCE_PX and p1_v[i] >= dynamic_min_threshold:
                    final_peaks[i] = p1_x[i]
                    status[i] = 1  # Case 1: Pure
                elif abs(p2_x[i] - median_1st_x) <= TOLERANCE_PX and p2_v[i] >= dynamic_min_threshold:
                    final_peaks[i] = p2_x[i]
                    status[i] = 2  # Case 2: Recovered
                else:
                    status[i] = 3  # Case 3: Missing
            
            # --- スプライン補間 (NaNの補完) ---
            peaks_series = pd.Series(final_peaks)
            if peaks_series.notna().sum() > 3: # スプライン補間には最低限のデータ数が必要
                corrected_boundary = peaks_series.interpolate(method='cubic').bfill().ffill().values
            else:
                corrected_boundary = peaks_series.interpolate(method='linear').bfill().ffill().values
            
            # --- 個別グラフのプロット (全体統計ベースでの分岐プロット) ---
            if not ONLY_SUMMARY:
                print(f"  -> 個別画像のプロットを生成中 ({f_name})...")
                for i in range(len(p1_x)):
                    b_name = data["base_filenames"][i]
                    dist = data["distances"][i]
                    a_deriv = data["abs_derivs"][i]
                    r_int = data["raw_intensities"][i]
                    f_titl = data["f_titles"][i]
                    stat = status[i]
                    c_bound = corrected_boundary[i]
                    p1_xi, p1_vi = p1_x[i], p1_v[i]
                    p2_xi, p2_vi = p2_x[i], p2_v[i]
                    
                    deriv_plot_path_raw = os.path.join(path_structure["1d_average"]["plots"], "derivative", "raw", f"deriv_{f_name}_{b_name}.png")
                    deriv_plot_path_corrected = os.path.join(path_structure["1d_average"]["plots"], "derivative", "corrected", f"deriv_{f_name}_{b_name}.png")
                    
                    # --- 1. Raw Plot (補間・検疫なしのありのままのピーク) ---
                    fig_raw, ax1_raw = plt.subplots(figsize=(10, 6))
                    ax1_raw.axvspan(0, START_DISTANCE, color='lightgray', alpha=0.5, label="Ignored Area")
                    ax1_raw.set_xlabel("Distance (pixels)")
                    ax1_raw.set_ylabel("Absolute Derivative", color="tab:orange")
                    line1_raw = ax1_raw.plot(dist, a_deriv, color="tab:orange", linewidth=2, label="Absolute Derivative")
                    peaks_plots_raw = ax1_raw.plot(p1_xi, p1_vi, "ro", markersize=8, label="Max Peak (Raw 1st Peak)")
                    ax1_raw.tick_params(axis='y', labelcolor="tab:orange")
                    ax1_raw.grid(True)
                    
                    ax2_raw = ax1_raw.twinx()
                    ax2_raw.set_ylabel("Intensity", color="tab:blue")
                    line2_raw = ax2_raw.plot(dist, r_int, color="tab:blue", linewidth=2, alpha=0.5, linestyle="--", label="Intensity")
                    ax2_raw.tick_params(axis='y', labelcolor="tab:blue")
                    
                    lines_raw = line1_raw + peaks_plots_raw + line2_raw
                    labels_raw = [l.get_label() for l in lines_raw]
                    ax1_raw.legend(lines_raw, labels_raw, loc="upper right")
                    plt.title(f"1D Derivative & Intensity: {f_titl} - {b_name} (Raw)")
                    plt.savefig(deriv_plot_path_raw, dpi=300); plt.close()

                    # --- 2. Corrected Plot (検疫と補間を反映したプロット) ---
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    ax1.axvspan(0, START_DISTANCE, color='lightgray', alpha=0.5, label="Ignored Area")
                    ax1.set_xlabel("Distance (pixels)")
                    ax1.set_ylabel("Absolute Derivative", color="tab:orange")
                    line1 = ax1.plot(dist, a_deriv, color="tab:orange", linewidth=2, label="Absolute Derivative")
                    
                    peaks_plots = []
                    title_suffix = ""
                    if stat == 1:
                        peaks_plots += ax1.plot(p1_xi, p1_vi, "ro", markersize=8, label="Max Peak (Case 1)")
                    elif stat == 2:
                        peaks_plots += ax1.plot(p1_xi, p1_vi, marker="x", color="tab:orange", markersize=8, linestyle="None", label="Rejected 1st Peak")
                        peaks_plots += ax1.plot(p2_xi, p2_vi, "go", markersize=8, label="Recovered 2nd Peak")
                    elif stat == 3:
                        title_suffix = " [Interpolated]"
                        peaks_plots += ax1.plot(p1_xi, p1_vi, marker="x", color="gray", markersize=8, linestyle="None", label="Rejected 1st Peak")
                        if p2_xi != p1_xi:
                            peaks_plots += ax1.plot(p2_xi, p2_vi, marker="x", color="gray", markersize=8, linestyle="None", label="Rejected 2nd Peak")
                        
                        idx_bound = int(round(c_bound))
                        val_bound = a_deriv[idx_bound] if 0 <= idx_bound < len(a_deriv) else 0
                        peaks_plots += ax1.plot(c_bound, val_bound, marker="o", color="tab:blue", markersize=8, linestyle="None", label="Interpolated Boundary")
                        ax1.axvline(median_1st_x, color="tab:blue", linestyle="--", alpha=0.5, label="Anchor X")

                    ax1.tick_params(axis='y', labelcolor="tab:orange")
                    ax1.grid(True)
                    
                    ax2 = ax1.twinx()
                    ax2.set_ylabel("Intensity", color="tab:blue")
                    line2 = ax2.plot(dist, r_int, color="tab:blue", linewidth=2, alpha=0.5, linestyle="--", label="Intensity")
                    ax2.tick_params(axis='y', labelcolor="tab:blue")
                    
                    lines = line1 + peaks_plots + line2
                    labels = [l.get_label() for l in lines]
                    ax1.legend(lines, labels, loc="upper right")
                    
                    plt.title(f"1D Derivative & Intensity: {f_titl} - {b_name}{title_suffix} (Corrected)")
                    plt.savefig(deriv_plot_path_corrected, dpi=300); plt.close()
            
            # ==================================
            # サマリー1: 補間なし (Raw 1st Peaks)
            # ==================================
            plt.figure(figsize=(12, 8))
            plt.imshow(intensities_matrix, aspect='auto', cmap='viridis')
            plt.colorbar(label='Normalized Gray Value (0.0 - 1.0)')
            
            plt.axvspan(0, START_DISTANCE, color='lightgray', alpha=0.5, label="Ignored Area")
            
            # ありのままの1次ピークを赤色の細い実線で繋ぐ
            plt.plot(p1_x, y_indices, color='red', linestyle='-', linewidth=1.5, marker='', label='Raw 1st Peaks')
            
            plt.title(f"All Angles Summary Heatmap: Raw 1st Peaks (Uncorrected) - {f_name}")
            plt.xlabel("Distance (pixels)")
            plt.ylabel("Image Index (Angle)")
            plt.legend()
            
            raw_summary_plot_path = os.path.join(path_structure["1d_average"]["plots"], "derivative", "raw", f"summary_all_angles_raw_peaks_{f_name}.png")
            plt.savefig(raw_summary_plot_path, dpi=300)
            plt.close()
            print(f"  -> Rawサマリー画像を保存しました: {raw_summary_plot_path}")

            # ==================================
            # サマリー2: 3色色分け＆補間線付き (Corrected)
            # ==================================
            plt.figure(figsize=(12, 8))
            plt.imshow(intensities_matrix, aspect='auto', cmap='viridis')
            plt.colorbar(label='Normalized Gray Value (0.0 - 1.0)')
            
            # 解析対象外エリア（0〜START_DISTANCE）に薄いグレーのマスクを敷く
            plt.axvspan(0, START_DISTANCE, color='lightgray', alpha=0.5, label="Ignored Area")
            
            # 下敷きとしてスプライン補間後のなめらかな最終境界線を白色の実線で繋ぐ
            plt.plot(corrected_boundary, y_indices, color='white', linestyle='-', linewidth=1, alpha=0.8, zorder=1)
            
            # 信頼度に基づく3色シグナル色分けプロット
            idx1 = (status == 1)
            idx2 = (status == 2)
            idx3 = (status == 3)
            
            if np.any(idx1):
                plt.scatter(corrected_boundary[idx1], y_indices[idx1], color='red', marker='o', s=20, label='Pure (Case 1)', zorder=2)
            if np.any(idx2):
                plt.scatter(corrected_boundary[idx2], y_indices[idx2], color='green', marker='o', s=20, label='Recovered (Case 2)', zorder=2)
            if np.any(idx3):
                plt.scatter(corrected_boundary[idx3], y_indices[idx3], color='blue', marker='x', s=30, label='Missing/Interpolated (Case 3)', zorder=2)
            
            plt.title(f"All Angles Summary Heatmap: {f_name} Normalized Intensity (Corrected)\n(Anchor X: {median_1st_x:.1f}, Thresh: {dynamic_min_threshold:.1f})")
            plt.xlabel("Distance (pixels)")
            plt.ylabel("Image Index (Angle)")
            plt.legend()
            
            corrected_summary_plot_path = os.path.join(path_structure["1d_average"]["plots"], "derivative", "corrected", f"summary_all_angles_corrected_3colors_{f_name}.png")
            plt.savefig(corrected_summary_plot_path, dpi=300)
            plt.close()
            print(f"  -> Correctedサマリー画像を保存しました: {corrected_summary_plot_path}")

    # 3. 機械学習ステップ（将来用）
    if RUN_ML_REGRESSION:
        print("\n[STEP 3] 機械学習モデルによる予測を実行中（未実装）...")
        pass

    print("\n--- 全ての処理が完了しました ---")

if __name__ == "__main__":
    main()