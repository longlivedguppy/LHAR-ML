import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

def apply_smoothing_filter(data, output_csv_path, method="gaussian"):
    """データのガタガタ（ノイズ）を滑らかにして保存する"""
    if method == "gaussian":
        # ガウシアンフィルタで平滑化（sigmaの値で滑らかさを調整できます）
        smoothed = gaussian_filter1d(data, sigma=5)
    
    # 次のステップのためにCSVとして保存
    df = pd.DataFrame({"Raw_Intensity": data, "Clean_Intensity": smoothed})
    df.to_csv(output_csv_path, index=False)
    
    return smoothed

def apply_median_filter(data, output_csv_path, kernel_size=5):
    """データの外れ値（スパイクノイズ）をメディアンフィルタで除去して保存する"""
    # フィルタのウィンドウサイズは奇数である必要があるための安全処理
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    smoothed = medfilt(data, kernel_size=kernel_size)
    
    df = pd.DataFrame({"Raw_Intensity": data, "Median_Intensity": smoothed})
    df.to_csv(output_csv_path, index=False)
    
    return smoothed

def apply_median_gaussian_filter(data, output_csv_path, kernel_size=5, sigma=5):
    """データの外れ値（スパイクノイズ）をメディアンフィルタで除去した後、ガウシアンフィルタで滑らかにして保存する"""
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    # 1. メディアンフィルタの適用
    median_smoothed = medfilt(data, kernel_size=kernel_size)
    # 2. ガウシアンフィルタの重ね掛け
    gaussian_smoothed = gaussian_filter1d(median_smoothed, sigma=sigma)
    
    df = pd.DataFrame({"Raw_Intensity": data, "Median_Intensity": median_smoothed, "Median_Gaussian_Intensity": gaussian_smoothed})
    df.to_csv(output_csv_path, index=False)
    
    return gaussian_smoothed