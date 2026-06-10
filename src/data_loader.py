import cv2
import numpy as np

def extract_multi_line_profile(img_path, coords, num_lines=5, line_gap=4, reduction="median"):
    """
    OpenCVで画像を読み込み、指定座標から右方向に複数本の輝度プロファイルを抽出する。
    中心のX座標をスタートとし、右に line_gap ずつずらしながら num_lines 本を抽出します。

    Args:
        img_path (str): 画像ファイルへのパス。
        coords (dict): 'x1', 'y1', 'y2' を含む辞書。
        num_lines (int): 抽出するラインの本数。
        line_gap (int): ライン間のピクセル数。
        reduction (str): 複数ラインを1本にまとめる方法 ('median' または 'mean')。

    Returns:
        tuple: (平均輝度プロファイル (1D array), 全ラインの輝度行列 (2D array))
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {img_path}")
    
    step = 1 if coords["y1"] <= coords["y2"] else -1
    y_vals = np.arange(coords["y1"], coords["y2"] + step, step)
    all_lines_data = []

    for i in range(num_lines):
        current_x = coords["x1"] + (i * line_gap)
        line_data = img[y_vals, current_x]
        all_lines_data.append(line_data)

    matrix_data = np.array(all_lines_data).T  # (height, num_lines)
    
    if reduction == "median":
        average_data = np.median(matrix_data, axis=1)
    else:
        average_data = np.mean(matrix_data, axis=1)
    
    return average_data, matrix_data