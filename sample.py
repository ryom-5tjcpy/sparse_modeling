import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

# 1. データ生成とノイズの追加
# -----------------------------------------------------

# データセットの特徴量数（合計100個）
N_SAMPLES = 50
N_FEATURES = 100
N_INFORMATIVE = 10  # 実際に目的変数に影響する特徴量は10個のみ

# 目的変数に影響を与える真の係数ベクトル（10個だけ非ゼロ）
true_coef = np.zeros(N_FEATURES)
true_coef[:N_INFORMATIVE] = np.linspace(1, 10, N_INFORMATIVE) # 最初の10個に重みを設定

# 説明変数Xと目的変数yを生成
np.random.seed(42)
X = np.random.randn(N_SAMPLES, N_FEATURES)
# 目的変数y = X * true_coef + ノイズ
y = X @ true_coef + np.random.randn(N_SAMPLES) * 5

# データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"訓練データ数: {X_train.shape[0]}, 特徴量数: {X_train.shape[1]}\n")

# 2. 線形回帰 (スパース性なし)
# -----------------------------------------------------
print("### 1. 線形回帰モデル (Linear Regression) ###")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_mse = mean_squared_error(y_test, lr.predict(X_test))
print(f"テストデータでのMSE: {lr_mse:.2f}")

# 3. LASSO回帰 (スパース性を導入)
# -----------------------------------------------------
# alpha (正則化パラメータλ) の値が大きいほど、より多くの係数がゼロになる
alpha_value = 10.0
print(f"\n### 2. LASSO回帰モデル (alpha={alpha_value}) ###")
lasso = Lasso(alpha=alpha_value, max_iter=10000)
lasso.fit(X_train, y_train)
lasso_mse = mean_squared_error(y_test, lasso.predict(X_test))
print(f"テストデータでのMSE: {lasso_mse:.2f}")

# 4. 係数（重み）の比較とスパース性の確認
# -----------------------------------------------------
print("\n### 3. 係数の比較（スパース性の確認） ###")

# 係数をデータフレームに格納
coef_df = pd.DataFrame({
    '特徴量ID': np.arange(N_FEATURES),
    '真の係数': true_coef,
    '線形回帰の係数': lr.coef_,
    'LASSO回帰の係数': lasso.coef_
})

# 非ゼロ係数の数をカウント
lr_non_zero = np.sum(lr.coef_ != 0)
lasso_non_zero = np.sum(lasso.coef_ != 0)

print(f"線形回帰の非ゼロ係数: {lr_non_zero} / {N_FEATURES} 個")
print(f"LASSO回帰の非ゼロ係数: {lasso_non_zero} / {N_FEATURES} 個")

# 比較のために上位と下位の係数値を表示 (表示を見やすくするために丸める)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', '{:.4f}'.format)
print("\n--- 係数テーブルの抜粋 ---")
print(coef_df[coef_df['真の係数'] > 0].head())
print("...")
print(coef_df[coef_df['真の係数'] == 0].tail())

# 5. 可視化
# -----------------------------------------------------
# 係数プロット 
plt.figure(figsize=(12, 6))
plt.plot(true_coef, 'o', label='真の係数 (True Coef.)')
plt.plot(lr.coef_, 'x', label='線形回帰 (Linear Reg.)')
plt.plot(lasso.coef_, '*', label=f'LASSO回帰 (alpha={alpha_value})')
plt.axvline(x=N_INFORMATIVE - 0.5, color='r', linestyle='--', label='重要特徴量の境界')
plt.title('真の係数 vs. 推定された係数の比較')
plt.xlabel('特徴量ID')
plt.ylabel('係数値')
plt.legend()
plt.grid(True)
plt.show()