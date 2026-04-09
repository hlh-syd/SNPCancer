import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, GridSearchCV

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

COLOR_SCHEMES = {1: ["#ffffd9","#edf8b1","#c7e9b4","#7fcdbb","#41b6c4","#1d91c0","#225ea8","#0c2c84"]}
STYLE_SCHEMES = {1: 'o'}
selected_colors = COLOR_SCHEMES[1]
selected_marker = STYLE_SCHEMES[1]

def simple_beeswarm(x_values, nbins=40, width=0.1):
    v = np.asarray(x_values).ravel()
    if v.size == 0:
        return np.zeros(0)
    rng = (np.min(v), np.max(v))
    if rng[0] == rng[1]:
        rng = (rng[0] - 0.1, rng[1] + 0.1)
    counts, edges = np.histogram(v, bins=nbins, range=rng)
    idx = np.digitize(v, edges) - 1
    idx = np.clip(idx, 0, nbins - 1)
    y = np.zeros_like(v, dtype=float)
    for b in range(nbins):
        mask = idx == b
        c = np.sum(mask)
        if c == 0:
            continue
        offs = np.linspace(-width, width, c)
        y[mask] = offs
    return y

def main():
    csv_path = os.path.join(os.getcwd(), 'simulation_data.csv')
    df = pd.read_csv(csv_path)
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    feature_names = df.columns[:-1].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    base = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    param_grid = {'n_estimators': [100, 200, 300]}
    gs = GridSearchCV(estimator=base, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=0, n_jobs=-1)
    gs.fit(X_train, y_train)
    model = gs.best_estimator_

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_int = explainer.shap_interaction_values(X_test)
    mean_signed_int = shap_int.mean(0)
    mean_abs_int = np.abs(mean_signed_int)

    abs_mean = np.mean(np.abs(shap_values), axis=0)
    top_k = min(10, len(feature_names))
    top_idx = np.argsort(abs_mean)[::-1][:top_k]
    top_features = [feature_names[i] for i in top_idx]

    cmap = mcolors.LinearSegmentedColormap.from_list("custom", selected_colors, N=256)
    norm_int = mcolors.Normalize(vmin=0, vmax=np.max(mean_abs_int) if np.max(mean_abs_int) > 0 else 1)

    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    ax_bee = fig.add_subplot(gs[0, 0])
    for j, fi in enumerate(top_idx):
        v = shap_values[:, fi]
        yj = simple_beeswarm(v, nbins=40, width=0.15)
        ax_bee.scatter(v, j + yj, s=10, c=v, cmap='RdBu_r', alpha=0.7)
    ax_bee.set_yticks(range(len(top_idx)))
    ax_bee.set_yticklabels(top_features)
    ax_bee.set_xlabel('SHAP value')
    ax_bee.set_title('Beeswarm (Top Features)')

    ax_mat = fig.add_subplot(gs[0, 1])
    n = len(top_idx)
    mat = mean_abs_int[np.ix_(top_idx, top_idx)]
    im = ax_mat.imshow(mat, cmap=cmap, norm=norm_int)
    ax_mat.set_xticks(range(n))
    ax_mat.set_yticks(range(n))
    ax_mat.set_xticklabels(top_features, rotation=45, ha='right')
    ax_mat.set_yticklabels(top_features)
    ax_mat.set_title('Interaction Strength (Abs Mean)')
    cb = fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)
    cb.set_label('Abs Mean Interaction')

    plt.tight_layout()
    out_path = os.path.join(os.getcwd(), 'shap_beeswarm_interaction_combined.png')
    plt.savefig(out_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()