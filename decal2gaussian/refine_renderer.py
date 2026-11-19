import cv2
import numpy as np

def refine_Image_merge(wall_path, shape_path, opacity=0.9, displace_strength=1.5, wear_amount=0.15):
    # 1. 画像読み込み（壁はBGR、図形はBGRAで読み込む）
    wall = cv2.imread(wall_path).astype(np.float32) / 255.0
    shape = cv2.imread(shape_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

    # サイズが異なる場合は図形を壁に合わせる
    if wall.shape[:2] != shape.shape[:2]:
        shape = cv2.resize(shape, (wall.shape[1], wall.shape[0]), interpolation=cv2.INTER_AREA)

    h, w = wall.shape[:2]
    wall_gray = cv2.cvtColor(wall, cv2.COLOR_BGR2GRAY)

    # 2. 壁の微細な凹凸に合わせて図形を数ピクセル歪ませる（Displace処理）
    # 細かいノイズ成分（高周波）と大きな凹凸を抽出
    blur_gray = cv2.GaussianBlur(wall_gray, (9, 9), 0)
    grad_x = cv2.Sobel(blur_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur_gray, cv2.CV_32F, 0, 1, ksize=3)

    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x += grad_x * displace_strength
    map_y += grad_y * displace_strength
    shape_displaced = cv2.remap(shape, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    shape_bgr = shape_displaced[:, :, :3]
    alpha = shape_displaced[:, :, 3]

    # 3. 溝の透過と「かすれ（経年劣化）」の生成
    # ① 壁の暗い溝（レンガの目地や深いヒビ）は塗料が乗らないようにマスクを削る
    crevice_mask = np.clip((wall_gray - 0.15) / 0.25, 0.0, 1.0)

    # ② 壁のザラつきを利用したランダムなかすれ（ペンキの剥がれ）を作る
    noise = np.random.normal(0.5, 0.2, (h, w)).astype(np.float32)
    # 壁の明暗とノイズを掛け合わせて不規則な剥がれマスクを生成
    wear_mask = np.clip((wall_gray * 0.7 + noise * 0.3) - wear_amount, 0.0, 1.0)
    wear_mask = np.clip(wear_mask * 2.5, 0.0, 1.0) # コントラスト強調

    # マスクを統合し、エッジをわずかにぼかしてデジタル特有の硬さを消す
    combined_alpha = alpha * crevice_mask * wear_mask
    final_alpha = cv2.GaussianBlur(combined_alpha, (3, 3), 0) * opacity
    final_alpha = np.expand_dims(final_alpha, axis=2)

    # 4. 壁の陰影・照明（ライティング）の転写
    # 壁の「局所的な明るさの変化」だけを抽出し、図形の色に掛け合わせる
    # これにより白や明るい色のペンキでも壁の凹凸・影が浮き出る
    local_mean = cv2.GaussianBlur(wall_gray, (101, 101), 0) + 1e-5
    lighting_map = np.expand_dims(wall_gray / local_mean, axis=2)
    lighting_map = np.clip(lighting_map, 0.45, 1.25)

    # 5. 色温度の馴染ませ（環境光の影響を少しだけ与える）
    # 壁全体の平均色を図形に10%だけブレンドして色浮きを防ぐ
    env_color = np.mean(wall, axis=(0, 1))
    harmonized_shape = shape_bgr * 0.9 + env_color * 0.1

    # 陰影を適用したペイント色
    painted_bgr = harmonized_shape * lighting_map

    # 6. 最終合成
    result = wall * (1.0 - final_alpha) + painted_bgr * final_alpha
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

    return result

# 実行例
# out = blend_paint_texture("wall.jpg", "shape.png", opacity=0.92, displace_strength=2.0, wear_amount=0.12)
# cv2.imwrite("painted_wall.jpg", out)
