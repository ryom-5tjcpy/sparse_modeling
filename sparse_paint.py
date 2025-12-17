import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
import matplotlib_fontja  # 先ほど導入した日本語化ライブラリ

# 1. テスト画像の作成 (シンプルなグラデーション画像)
# -----------------------------------------------------
img_size = (64, 64)
y, x = np.ogrid[:img_size[0], :img_size[1]]
image = np.sin(x / 10) + np.cos(y / 10)
image = (image - image.min()) / (image.max() - image.min()) # 0-1に正規化

# 2. 画像の一部をランダムに欠損させる (50%の画素を失わせる)
# -----------------------------------------------------
missing_mask = np.random.choice([0, 1], size=img_size, p=[0.5, 0.5])
corrupted_image = image * missing_mask

# 3. スパースモデリングによる復元準備
# -----------------------------------------------------
# 画像を 8x8 のパッチに分割
patch_size = (8, 8)
patches = extract_patches_2d(corrupted_image, patch_size)
# 1次元ベクトルに変換
data = patches.reshape(patches.shape[0], -1)

# 学習用辞書の作成 (ここでは簡易的にDCT基底のようなランダム直交基底を使用)
# 本来は学習データから「辞書学習」を行いますが、ここではランダム基底で代用
n_components = 100 
dictionary = np.random.randn(data.shape[1], n_components)
dictionary /= np.linalg.norm(dictionary, axis=0) # 正規化

# 4. 各パッチに対してLASSOでスパース表現を求める
# -----------------------------------------------------
# 「少ない基底の組み合わせで、欠損していない画素を説明する」
reconstructed_patches = []
lasso = Lasso(alpha=0.01, max_iter=1000)

print("復元処理中...")
for d in data[:500]: # 計算短縮のため一部のパッチのみ処理
    # 欠損していない部分のインデックスを取得
    mask = d != 0
    if np.sum(mask) > 0:
        # 欠損していない画素だけを使って、辞書の重みを推定
        lasso.fit(dictionary[mask], d[mask])
        # 推定された重みを使ってフルパッチを再現
        reconstructed_patches.append(dictionary @ lasso.coef_)
    else:
        reconstructed_patches.append(d)

# 残りのパッチは元のままにする（デモ用簡略化）
reconstructed_patches = np.array(reconstructed_patches)
full_reconstructed_patches = np.vstack([reconstructed_patches, data[500:]])
full_reconstructed_patches = full_reconstructed_patches.reshape(-1, *patch_size)

# パッチを結合して画像に戻す
reconstructed_image = reconstruct_from_patches_2d(full_reconstructed_patches, img_size)

# 5. 結果の可視化
# -----------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("1. オリジナル画像")
axes[1].imshow(corrupted_image, cmap='gray')
axes[1].set_title("2. 欠損した画像 (50% Loss)")
axes[2].imshow(reconstructed_image, cmap='gray')
axes[2].set_title("3. スパース復元後 (LASSO使用)")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()