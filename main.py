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
    print("処理モードを選択してください [1: 生画像から処理 (raw), 2: 既存のクリーンデータを使用 (processed)]: ")
    mode = input("> ")

    if mode == '1':
        scan_dir = "data/raw"
    elif mode == '2':
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
    filter_types = ["median", "gaussian", "median_gaussian", "derivative"]
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
        
    elif mode == '2':
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

        # --- モード2: 既存のフィルタ済みCSVからデータ読み込み ---
        elif mode == '2':
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
                
                # scipy.signal.find_peaks を用いて最も顕著なピークを自動検出
                peaks, properties = find_peaks(abs_derivative_clipped, height=0)
                if len(peaks) > 0:
                    best_peak_idx = peaks[np.argmax(properties['peak_heights'])]
                else:
                    best_peak_idx = np.argmax(abs_derivative_clipped) # フォールバック
                best_peak_val = abs_derivative[best_peak_idx]
                
                # --- サマリー用の2D配列とピーク位置をメモリに格納 ---
                if f_name not in summary_data:
                    summary_data[f_name] = {"intensities": [], "peaks": []}
                summary_data[f_name]["intensities"].append(intensities)
                summary_data[f_name]["peaks"].append(best_peak_idx)
                
                df_deriv = pd.DataFrame({"Distance": distance, "Intensity": intensities, "Derivative": derivative})
                df_deriv.to_csv(deriv_csv_path, index=False)
                
                plt.figure(figsize=(10, 6))
                # 解析対象外エリア（0〜START_DISTANCE）に薄いグレーの背景を敷く
                plt.axvspan(0, START_DISTANCE, color='lightgray', alpha=0.5, label="Ignored Area")
                
                plt.plot(distance, abs_derivative, color="orange", linewidth=2, label="Absolute Derivative")
                # 最も顕著なピークを赤点でマーキング
                plt.plot(distance[best_peak_idx], best_peak_val, "ro", markersize=8, label=f"Max Peak (Distance={distance[best_peak_idx]})")
                
                plt.title(f"1D Derivative: {f_title} - {base_filename}")
                plt.xlabel("Distance (pixels)"); plt.ylabel("Absolute Derivative"); plt.legend(); plt.grid(True)
                plt.savefig(deriv_plot_path, dpi=300); plt.close()

    # ==========================================
    # 全角度微分サマリーヒートマップの生成と保存
    # ==========================================
    if RUN_DERIVATIVE_ANALYSIS and summary_data:
        print("\n[サマリー生成] 全角度サマリーヒートマップを生成・保存中...")
        for f_name, data in summary_data.items():
            intensities_matrix = np.array(data["intensities"])  # Shape: [num_images, 761]
            peaks_array = np.array(data["peaks"])     # Shape: [num_images]
            y_indices = np.arange(len(peaks_array))
            
            plt.figure(figsize=(12, 8))
            # aspect='auto' とすることで、行数(61)と列数(761)の比率をよしなに画面幅に合わせてくれます
            plt.imshow(intensities_matrix, aspect='auto', cmap='viridis')
            plt.colorbar(label='Gray Value')
            
            # 解析対象外エリア（0〜START_DISTANCE）に薄いグレーのマスクを敷く
            plt.axvspan(0, START_DISTANCE, color='lightgray', alpha=0.5, label="Ignored Area")
            
            # 検出した各ピークのDistance位置を赤線（点群）として重ねる
            plt.plot(peaks_array, y_indices, color='red', marker='o', markersize=3, linestyle='-', linewidth=1.5, label='Detected Boundary (Peaks)')
            
            plt.title(f"All Angles Summary Heatmap: {f_name} Intensity")
            plt.xlabel("Distance (pixels)")
            plt.ylabel("Image Index (Angle)")
            plt.legend()
            
            summary_plot_path = os.path.join(path_structure["1d_average"]["plots"], "derivative", f"summary_all_angles_intensity_{f_name}.png")
            plt.savefig(summary_plot_path, dpi=300)
            plt.close()
            print(f"  -> サマリー画像を保存しました: {summary_plot_path}")

    # 3. 機械学習ステップ（将来用）
    if RUN_ML_REGRESSION:
        print("\n[STEP 3] 機械学習モデルによる予測を実行中（未実装）...")
        pass

    print("\n--- 全ての処理が完了しました ---")

if __name__ == "__main__":
    main()