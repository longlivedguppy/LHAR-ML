import os
import glob
import numpy as np
import matplotlib.pyplot as plt
# srcフォルダ内の自作モジュールから関数をインポート
from src.data_loader import extract_multi_line_profile
from src.denoise import apply_smoothing_filter, apply_median_filter, apply_median_gaussian_filter
# 2D行列のフィルタリングにScipy関数を直接使用
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

def main():
    # ==========================================
    # 解析・MLの実行コントロールスイッチ (True/False)
    # ==========================================
    RUN_DATA_EXTRACTION = True   # 生画像からライン輝度データを抽出する
    RUN_MEDIAN_FILTER   = False   # メディアンフィルタによるスパイクノイズ除去を行う
    RUN_GAUSSIAN_FILTER = False   # ガウシアンフィルタによる平滑化を行う
    RUN_MEDIAN_GAUSSIAN_FILTER = True # メディアンフィルタ適用後にガウシアンフィルタをかける
    RUN_ML_REGRESSION   = False  # 【今後実装】機械学習による膜厚・深さ予測
    RUN_EDGE_DETECTION  = False  # 【今後実装】製膜境界（端）の自動検出

    # ==========================================
    # 処理パラメータとパスの設定
    # ==========================================
    # 処理対象とする【実験データの撮影日フォルダ名】を指定（今日の日付ではありません）
    target_dataset_dir = "20260512"
    N_LINES = 5
    LINE_GAP = 4

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
    filter_types = ["median", "gaussian", "median_gaussian"]
    for proc_type, paths in path_structure.items():
        for category, base_path in paths.items():
            for filter_type in filter_types:
                os.makedirs(os.path.join(base_path, filter_type), exist_ok=True)

    # ==========================================
    # 処理対象の画像ファイルリストを取得
    # ==========================================
    # .tiffと.tifの両方に対応
    img_files = glob.glob(os.path.join(raw_dir, "*.tiff")) + glob.glob(os.path.join(raw_dir, "*.tif"))
    if not img_files:
        print(f"エラー: {raw_dir} 内に処理対象の画像ファイル（.tiff, .tif）が見つかりません。")
        return

    print(f"--- {target_dataset_dir} フォルダ内の {len(img_files)} 個の画像を処理します ---")

    # ==========================================
    # 🏁 実際の解析処理の実行パート (ループ処理)
    # ==========================================
    progress_bar = tqdm(img_files, unit="file", desc="Image Processing")
    for img_path in progress_bar:
        base_filename_with_ext = os.path.basename(img_path)
        # プログレスバーの右側に現在のファイル名を表示
        progress_bar.set_postfix_str(f"{base_filename_with_ext}", refresh=True)

        # 出力ファイル名の生成
        base_filename = os.path.splitext(base_filename_with_ext)[0]

        avg_intensities = None  # 各ファイル処理の前に初期化
        matrix_intensities = None

        # 1. データ抽出ステップ
        if RUN_DATA_EXTRACTION:
            try:
                avg_intensities, matrix_intensities = extract_multi_line_profile(
                    img_path, roi_coords, num_lines=N_LINES, line_gap=LINE_GAP)
            except FileNotFoundError as e:
                tqdm.write(f"\nエラー: {e}")
                continue  # 次のファイルへ
            except Exception as e:
                tqdm.write(f"\n予期せぬエラーが発生しました: {e}")
                continue  # 次のファイルへ
        
        # 抽出データがない場合は後続のフィルタ処理をスキップ
        if avg_intensities is None and (RUN_MEDIAN_FILTER or RUN_GAUSSIAN_FILTER or RUN_MEDIAN_GAUSSIAN_FILTER):
                tqdm.write(f"\nスキップ ({base_filename_with_ext}): 輝度データがありません。RUN_DATA_EXTRACTIONをTrueにしてください。")
                continue # 次のファイルへ

        # 2-A. メディアンフィルタ処理（外れ値・スパイクノイズ除去）
        if RUN_MEDIAN_FILTER:
            # --- 1D Processing (Average) ---
            csv_path_1d = os.path.join(path_structure["1d_average"]["data"], "median", f"median_{base_filename}.csv")
            plot_path_1d = os.path.join(path_structure["1d_average"]["plots"], "median", f"median_{base_filename}.png")
            median_intensities = apply_median_filter(avg_intensities, csv_path_1d, kernel_size=5)
            plt.figure(figsize=(10, 6))
            plt.plot(avg_intensities, color="gray", alpha=0.5, label="Raw Average")
            plt.plot(median_intensities, color="blue", linewidth=2, label="Median Filtered Data")
            plt.title(f"1D Average: Median Filter - {base_filename_with_ext}")
            plt.xlabel("Distance (pixels)"); plt.ylabel("Gray Value"); plt.legend(); plt.grid(True)
            plt.savefig(plot_path_1d, dpi=300); plt.close()

            # --- 2D Processing (Surface Plot) ---
            plot_path_2d = os.path.join(path_structure["2d_surface"]["plots"], "median", f"surface_median_{base_filename}.png")
            filtered_matrix = np.apply_along_axis(medfilt, 0, matrix_intensities, kernel_size=5)
            plt.figure(figsize=(8, 6))
            plt.imshow(filtered_matrix, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(label='Gray Value')
            plt.title(f"2D Surface: Median Filter - {base_filename_with_ext}")
            plt.xlabel("Line Number"); plt.ylabel("Distance (pixels)")
            plt.savefig(plot_path_2d, dpi=300); plt.close()
        
        # 2-B. ガウシアンフィルタ処理（全体的な平滑化）
        if RUN_GAUSSIAN_FILTER:
            # --- 1D Processing (Average) ---
            csv_path_1d = os.path.join(path_structure["1d_average"]["data"], "gaussian", f"gaussian_{base_filename}.csv")
            plot_path_1d = os.path.join(path_structure["1d_average"]["plots"], "gaussian", f"gaussian_{base_filename}.png")
            clean_intensities = apply_smoothing_filter(avg_intensities, csv_path_1d, method="gaussian")
            plt.figure(figsize=(10, 6))
            plt.plot(avg_intensities, color="gray", alpha=0.5, label="Raw Average")
            plt.plot(clean_intensities, color="black", linewidth=2, label="Gaussian Filtered Data")
            plt.title(f"1D Average: Gaussian Filter - {base_filename_with_ext}")
            plt.xlabel("Distance (pixels)"); plt.ylabel("Gray Value"); plt.legend(); plt.grid(True)
            plt.savefig(plot_path_1d, dpi=300); plt.close()

            # --- 2D Processing (Surface Plot) ---
            plot_path_2d = os.path.join(path_structure["2d_surface"]["plots"], "gaussian", f"surface_gaussian_{base_filename}.png")
            filtered_matrix = np.apply_along_axis(gaussian_filter1d, 0, matrix_intensities, sigma=5)
            plt.figure(figsize=(8, 6))
            plt.imshow(filtered_matrix, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(label='Gray Value')
            plt.title(f"2D Surface: Gaussian Filter - {base_filename_with_ext}")
            plt.xlabel("Line Number"); plt.ylabel("Distance (pixels)")
            plt.savefig(plot_path_2d, dpi=300); plt.close()
            
        # 2-C. ハイブリッドフィルタ処理（メディアン → ガウシアン）
        if RUN_MEDIAN_GAUSSIAN_FILTER:
            # --- 1D Processing (Average) ---
            csv_path_1d = os.path.join(path_structure["1d_average"]["data"], "median_gaussian", f"median_gaussian_{base_filename}.csv")
            plot_path_1d = os.path.join(path_structure["1d_average"]["plots"], "median_gaussian", f"median_gaussian_{base_filename}.png")
            hybrid_intensities = apply_median_gaussian_filter(avg_intensities, csv_path_1d, kernel_size=5, sigma=5)
            plt.figure(figsize=(10, 6))
            plt.plot(avg_intensities, color="gray", alpha=0.5, label="Raw Average")
            plt.plot(hybrid_intensities, color="red", linewidth=2, label="Median + Gaussian Filtered Data")
            plt.title(f"1D Average: Hybrid Filter - {base_filename_with_ext}")
            plt.xlabel("Distance (pixels)"); plt.ylabel("Gray Value"); plt.legend(); plt.grid(True)
            plt.savefig(plot_path_1d, dpi=300); plt.close()

            # --- 2D Processing (Surface Plot) ---
            plot_path_2d = os.path.join(path_structure["2d_surface"]["plots"], "median_gaussian", f"surface_median_gaussian_{base_filename}.png")
            median_filtered_matrix = np.apply_along_axis(medfilt, 0, matrix_intensities, kernel_size=5)
            hybrid_filtered_matrix = np.apply_along_axis(gaussian_filter1d, 0, median_filtered_matrix, sigma=5)
            plt.figure(figsize=(8, 6))
            plt.imshow(hybrid_filtered_matrix, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(label='Gray Value')
            plt.title(f"2D Surface: Hybrid Filter - {base_filename_with_ext}")
            plt.xlabel("Line Number"); plt.ylabel("Distance (pixels)")
            plt.savefig(plot_path_2d, dpi=300); plt.close()
        
    # 3. 機械学習ステップ（将来用）
    if RUN_ML_REGRESSION:
        print("\n[STEP 3] 機械学習モデルによる予測を実行中（未実装）...")
        pass

    print("\n--- 全ての処理が完了しました ---")

if __name__ == "__main__":
    main()