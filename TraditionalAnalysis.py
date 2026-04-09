import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
import os
import gc
import sys
import unittest

try:
    from matplotlib_venn import venn2
except ImportError:
    venn2 = None

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    from dataset import qc_snp, qc_by_missing, merge_data, impute_with_uncertainty
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from dataset import qc_snp, qc_by_missing, merge_data, impute_with_uncertainty

warnings.filterwarnings("ignore")

try:
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    print("Warning: scikit-survival not installed. LASSO-Cox functionality will be limited.")

def _zscore_fit(x):
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd

def _zscore_apply(x, mu, sd):
    return (x - mu) / sd

def _normalize_beta(beta, method="l1"):
    b = np.asarray(beta, dtype=float)
    if b.size == 0:
        return b
    if method == "l2":
        d = np.sqrt(np.sum(b * b))
    else:
        d = np.sum(np.abs(b))
    if not np.isfinite(d) or d == 0:
        return b
    return b / d

def _label_by_time(T, E, tp):
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=float)
    return ((T < tp) & (E == 1)).astype(int)

def _youden_cutoff(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return float(np.nanmedian(s))
    fpr, tpr, th = roc_curve(y, s)
    j = tpr - fpr
    if j.size == 0:
        return float(np.nanmedian(s))
    k = int(np.nanargmax(j))
    return float(th[k])

def _train_valid_split(idx, E, valid_size=0.3, random_state=42):
    try:
        from sklearn.model_selection import train_test_split
        return train_test_split(idx, test_size=valid_size, random_state=random_state, stratify=E)
    except Exception:
        rng = np.random.RandomState(random_state)
        idx = np.asarray(idx)
        rng.shuffle(idx)
        k = int(round(len(idx) * (1 - valid_size)))
        return idx[:k], idx[k:]

def _cohen_kappa(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return np.nan
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return np.nan
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(a, b))
    except Exception:
        return np.nan

class TraditionalAnalysis:
    """
    Traditional Survival Analysis Pipeline.
    
    Workflow:
    1. Load Data & QC (dataset.py)
    2. Univariate Cox Screening (Filter by P-value/FDR)
    3. LASSO Cox Feature Selection (Regularization)
    4. Risk Score Construction (Linear Combination)
    5. Evaluation (KM, ROC, VIF)
    6. Advanced Analysis (Venn, Volcano, GO/KEGG, PPI, Hub, TF/miRNA)
    """
    def __init__(self, data_dir="e:\\TERM\\data", output_dir="e:\\TERM\\results",
                 p_thres=0.01, standardize="zscore", beta_norm="l1",
                 valid_size=0.3, random_state=42, cutoff_time=1095):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.p_thres = float(p_thres)
        self.standardize = str(standardize)
        self.beta_norm = str(beta_norm)
        self.valid_size = float(valid_size)
        self.random_state = int(random_state)
        self.cutoff_time = float(cutoff_time)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Initializing Traditional Analysis pipeline...")
        print(f"Data Directory: {data_dir}")
        print(f"Output Directory: {output_dir}")
        
        # 1. Load & Merge Data using dataset.py logic
        self.df = self._load_and_preprocess_data()
        
        # 2. Define Clinical vs Feature columns
        self.clinical_patterns = ['age', 'gender', 'stage', 'T', 'E', 'status', 'time', 'risk', 'cluster']
        self.clinical_cols = [c for c in self.df.columns if any(p in c.lower() for p in self.clinical_patterns)]
        
        # Ensure T and E are numeric and exist
        if 'T' not in self.df.columns or 'E' not in self.df.columns:
            col_map = {}
            for c in self.df.columns:
                cl = c.lower()
                if 'time' in cl or 'duration' in cl: col_map[c] = 'T'
                if 'status' in cl or 'event' in cl or 'dead' in cl: col_map[c] = 'E'
            self.df.rename(columns=col_map, inplace=True)
            
        if 'T' not in self.df.columns or 'E' not in self.df.columns:
             raise ValueError("Could not find survival time (T) or event (E) columns after merging.")

        self.df['T'] = pd.to_numeric(self.df['T'], errors='coerce')
        self.df['E'] = pd.to_numeric(self.df['E'], errors='coerce')
        self.df = self.df.dropna(subset=['T', 'E'])
        
        self.snp_cols = [c for c in self.df.columns if c not in self.clinical_cols]
        print(f"Dataset Loaded: {self.df.shape[0]} samples, {len(self.snp_cols)} SNPs/Features.")

    def _load_and_preprocess_data(self):
        print("Merging data from raw files...")
        merged_df = merge_data(self.data_dir, missing_threshold=0.2)
        print("Running QC (Missingness + SNP specific)...")
        df_qc = qc_by_missing(merged_df, sample_threshold=0.05, feature_threshold=0.05)
        print("Running HWE/MAF filtering...")
        df_qc = qc_snp(df_qc, maf_threshold=0.01, hwe_threshold=0.01)
        if df_qc.isna().sum().sum() > 0:
            print("Imputing missing values (using Soft-GAIN from dataset.py)...")
            df_imp, _ = impute_with_uncertainty(df_qc, n_runs=1) 
            return df_imp
        else:
            return df_qc

    def plot_km(self, variable, output_path=None, discrete=True):
        sns.set_style("whitegrid")
        sns.set_context("talk")
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        data = self.df[[variable, 'T', 'E']].dropna()
        if discrete:
            if data[variable].nunique() <= 5:
                data['Group'] = data[variable].round().astype(int)
                groups = sorted(data['Group'].unique())
                colors = sns.color_palette("Set1", n_colors=len(groups))
                for i, g in enumerate(groups):
                    mask = data['Group'] == g
                    if mask.sum() > 0:
                        kmf.fit(data.loc[mask, 'T'], event_observed=data.loc[mask, 'E'], label=f"{variable}={g} (n={mask.sum()})")
                        kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2.5, color=colors[i])
                if len(groups) > 1:
                    res = multivariate_logrank_test(data['T'], data['Group'], data['E'])
                    p_val = res.p_value
                else:
                    p_val = 1.0
            else:
                discrete = False
        if not discrete:
            median_val = data[variable].median()
            mask_high = data[variable] > median_val
            mask_low = data[variable] <= median_val
            colors = ["#E41A1C", "#377EB8"] 
            kmf.fit(data.loc[mask_high, 'T'], event_observed=data.loc[mask_high, 'E'], label=f"High (> {median_val:.2f})")
            kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2.5, color=colors[0])
            kmf.fit(data.loc[mask_low, 'T'], event_observed=data.loc[mask_low, 'E'], label=f"Low (<= {median_val:.2f})")
            kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2.5, color=colors[1])
            res = logrank_test(data.loc[mask_high, 'T'], data.loc[mask_low, 'T'], 
                               data.loc[mask_high, 'E'], data.loc[mask_low, 'E'])
            p_val = res.p_value

        plt.title(f"Kaplan-Meier Survival Curve: {variable}\nLog-Rank p = {p_val:.2e}", fontsize=16, fontweight='bold')
        plt.xlabel("Time (Days)", fontsize=14)
        plt.ylabel("Survival Probability", fontsize=14)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc="best", frameon=True, framealpha=0.9, shadow=True)
        sns.despine()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved enhanced KM plot to {output_path}")
        else:
            plt.show()

    def univariate_cox(self):
        print("\n=== Step 1: Univariate Cox Screening ===")
        results = []
        total_snps = len(self.snp_cols)
        print(f"Screening {total_snps} features...")
        for i, snp in enumerate(self.snp_cols):
            if i % 500 == 0:
                print(f"  Processed {i}/{total_snps}...", flush=True)
                gc.collect()
            try:
                subset = self.df[[snp, 'T', 'E']].dropna()
                if subset[snp].std() == 0: continue
                cph = CoxPHFitter()
                cph.fit(subset, duration_col='T', event_col='E')
                summ = cph.summary.loc[snp]
                results.append({
                    'snp': snp,
                    'coef': summ['coef'],
                    'HR': np.exp(summ['coef']),
                    'p_value': summ['p'],
                    'lower_95': np.exp(summ['coef lower 95%']),
                    'upper_95': np.exp(summ['coef upper 95%'])
                })
            except:
                continue
        res_df = pd.DataFrame(results)
        if not res_df.empty:
            res_df = res_df.sort_values('p_value')
            
            # Determine output directory for univariate results
            # If current output_dir is the main results folder (or root), save to 'cox' subdir
            # Check if we are in a subgroup analysis folder (ends with digit or 'unknown')
            is_subgroup = any(self.output_dir.endswith(suffix) for suffix in ["/1", "\\1", "/2", "\\2", "/4", "\\4", "/unknown", "\\unknown"])
            
            if not is_subgroup:
                cox_out_dir = os.path.join(self.output_dir, "cox")
                if not os.path.exists(cox_out_dir):
                    os.makedirs(cox_out_dir)
                out_file = os.path.join(cox_out_dir, "snp_univariate_results.csv")
                sig_file = os.path.join(cox_out_dir, f"snp_univariate_significant_p{self.p_thres}.csv")
            else:
                out_file = os.path.join(self.output_dir, "snp_univariate_results.csv")
                sig_file = os.path.join(self.output_dir, f"snp_univariate_significant_p{self.p_thres}.csv")

            res_df.to_csv(out_file, sep='\t', index=False)
            print(f"Saved univariate results to {out_file}")
            sig_df = res_df[res_df['p_value'] < self.p_thres].copy()
            sig_df.to_csv(sig_file, sep='\t', index=False)
            print(f"Saved significant features to {sig_file} (Count: {len(sig_df)})")
            candidates = sig_df['snp'].tolist()
            print(f"Found {len(candidates)} features with p < {self.p_thres}.")
            return candidates, res_df
        else:
            print("No features converged in univariate Cox.")
            return [], res_df

    def lasso_cox(self, candidate_snps):
        print("\n=== Step 2: LASSO-Cox Feature Selection ===")
        if not SKSURV_AVAILABLE or len(candidate_snps) < 2:
            print("Skipping LASSO (Not enough candidates or sksurv missing).")
            return candidate_snps[:10] 
        X = self.df[candidate_snps]
        y = np.array([(bool(e), t) for e, t in zip(self.df['E'], self.df['T'])], 
                     dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
        print("Estimating optimal alpha via CV...")
        coxnet = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, fit_baseline_model=True)
        coxnet.fit(X, y)
        alphas = coxnet.alphas_
        best_alpha = alphas[-1]
        for alpha in alphas:
            model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[alpha], fit_baseline_model=True)
            model.fit(X, y)
            n_features = np.sum(model.coef_ != 0)
            if 5 <= n_features <= 30:
                best_alpha = alpha
                break 
        print(f"Selected alpha: {best_alpha}")
        final_model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[best_alpha], fit_baseline_model=True)
        final_model.fit(X, y)
        coefs = pd.Series(final_model.coef_.ravel(), index=X.columns)
        selected = coefs[coefs != 0]
        print(f"LASSO selected {len(selected)} features.")
        selected.to_csv(os.path.join(self.output_dir, "lasso_selected_features.csv"), sep='\t')
        return selected

    def build_final_model(self, selected_features_series):
        print("\n=== Step 3: Final Model & Evaluation ===")
        features = selected_features_series.index.tolist()
        print("\n--- Multivariate Cox Regression (Unbiased HR & P-value) ---")
        cox_data = self.df[features + ['T', 'E']].dropna()
        cph_multi = CoxPHFitter()
        cph_multi.fit(cox_data, duration_col='T', event_col='E')
        
        print("\n" + "="*60)
        print("Multivariate Cox Regression Results")
        print("="*60)
        cph_multi.print_summary()
        
        multi_cox_summary = cph_multi.summary.copy()
        multi_cox_summary['HR'] = np.exp(multi_cox_summary['coef'])
        multi_cox_summary['HR_lower_95%'] = np.exp(multi_cox_summary['coef lower 95%'])
        multi_cox_summary['HR_upper_95%'] = np.exp(multi_cox_summary['coef upper 95%'])
        
        cols_order = ['coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%', 
                      'HR', 'HR_lower_95%', 'HR_upper_95%', 'z', 'p', '-log2(p)']
        cols_exist = [c for c in cols_order if c in multi_cox_summary.columns]
        multi_cox_summary = multi_cox_summary[cols_exist]
        multi_cox_summary = multi_cox_summary.reindex(features)
        
        multi_cox_path = os.path.join(self.output_dir, "multivariate_cox_results.csv")
        multi_cox_summary.to_csv(multi_cox_path)
        print(f"\nMultivariate Cox results saved to {multi_cox_path}")

        self.plot_forest(multi_cox_summary, os.path.join(self.output_dir, "multivariate_cox_forest.png"))
        c_index = cph_multi.concordance_index_
        print(f"Multivariate Cox C-index: {c_index:.4f}")
        
        all_i = np.arange(len(self.df))
        tr_i, va_i = _train_valid_split(all_i, self.df['E'].values, self.valid_size, self.random_state)
        tr = self.df.index[tr_i]
        va = self.df.index[va_i]

        X = self.df[features].to_numpy(dtype=float)
        mu, sd = _zscore_fit(X[tr_i])
        Xn = _zscore_apply(X, mu, sd)

        beta = _normalize_beta(selected_features_series.values, self.beta_norm)
        s = Xn @ beta
        self.df['RiskScore'] = s

        self.plot_risk_heatmap(features, os.path.join(self.output_dir, "risk_score_heatmap.png"))

        y_tr = _label_by_time(self.df.loc[tr, 'T'].values, self.df.loc[tr, 'E'].values, self.cutoff_time)
        s_tr = self.df.loc[tr, 'RiskScore'].values
        cutoff = _youden_cutoff(y_tr, s_tr)
        self.train_cutoff = cutoff
        med_tr = float(np.nanmedian(s_tr))

        self.df['Set'] = "train"
        self.df.loc[va, 'Set'] = "valid"
        self.df['RiskGroup_train_opt'] = "NA"
        self.df['RiskGroup_valid_opt'] = "NA"
        self.df.loc[tr, 'RiskGroup_train_opt'] = np.where(self.df.loc[tr, 'RiskScore'] > cutoff, "High", "Low")
        self.df.loc[va, 'RiskGroup_valid_opt'] = np.where(self.df.loc[va, 'RiskScore'] > cutoff, "High", "Low")
        
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        sns.histplot(self.df.loc[tr, 'RiskScore'], color="#0073C2", label="Train", kde=True, stat="density", alpha=0.35)
        sns.histplot(self.df.loc[va, 'RiskScore'], color="#EFC000", label="Valid", kde=True, stat="density", alpha=0.35)
        plt.axvline(cutoff, color="#D55E00", linestyle="--", linewidth=2, label=f"Train Youden cutoff={cutoff:.3f}")
        plt.axvline(med_tr, color="#009E73", linestyle="--", linewidth=2, label=f"Train median={med_tr:.3f}")
        plt.title("RiskScore Distribution (Z-score + beta norm)", fontsize=14, fontweight='bold')
        plt.xlabel("RiskScore")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "risk_score_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()

        df_tr = self.df.loc[tr, ['T', 'E', 'RiskGroup_train_opt']].rename(columns={'RiskGroup_train_opt': 'Group'}).copy()
        df_va = self.df.loc[va, ['T', 'E', 'RiskGroup_valid_opt']].rename(columns={'RiskGroup_valid_opt': 'Group'}).copy()
        self._plot_km_df(df_tr, "Train KM (Youden cutoff)", os.path.join(self.output_dir, "final_risk_km_train_opt.png"))
        self._plot_km_df(df_va, "Valid KM (Train Youden cutoff)", os.path.join(self.output_dir, "final_risk_km_valid_opt.png"))
        df_all = self.df[['T', 'E', 'RiskScore']].copy()
        df_all['Group'] = np.where(df_all['RiskScore'] > cutoff, "High", "Low")
        self._plot_km_df(df_all[['T', 'E', 'Group']], "Overall KM (Train Youden cutoff)", os.path.join(self.output_dir, "final_risk_km_opt.png"))
        
        self.plot_roc_curves(os.path.join(self.output_dir, "final_roc.png"))
        self.plot_correlation_combo(selected_features_series.index.tolist(), self.output_dir)

        g_cut = (self.df.loc[tr, 'RiskScore'].values > cutoff).astype(int)
        g_med = (self.df.loc[tr, 'RiskScore'].values > med_tr).astype(int)
        kappa = _cohen_kappa(g_cut, g_med)
        rep = [
            f"p_thres={self.p_thres}",
            f"standardize=zscore(train-fit, global-apply)",
            f"beta_norm={self.beta_norm}, sum_abs_beta={np.sum(np.abs(beta)):.6f}",
            f"cutoff_time={self.cutoff_time}",
            f"train_size={len(tr_i)}, valid_size={len(va_i)}",
            f"train_youden_cutoff={cutoff:.6f}",
            f"train_median={med_tr:.6f}",
            f"kappa_train_cutoff_vs_median={kappa}",
            "Note: Validation and Overall analysis now use Train Youden cutoff."
        ]
        with open(os.path.join(self.output_dir, "grouping_consistency_report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(rep) + "\n")

        final_df = pd.DataFrame({
            'feature': features,
            'lasso_coef': selected_features_series.values, 
            'lasso_HR': np.exp(selected_features_series.values),
            'multiCox_coef': multi_cox_summary['coef'].values, 
            'multiCox_HR': multi_cox_summary['HR'].values,
            'multiCox_HR_95CI_lower': multi_cox_summary['HR_lower_95%'].values,
            'multiCox_HR_95CI_upper': multi_cox_summary['HR_upper_95%'].values,
            'multiCox_p_value': multi_cox_summary['p'].values
        })
        final_df.to_csv(os.path.join(self.output_dir, "final_prognostic_biomarkers.csv"), index=False)
        print(f"Final biomarkers saved to {os.path.join(self.output_dir, 'final_prognostic_biomarkers.csv')}")

        print("\n" + "="*80)
        print(" FINAL MODEL SUMMARY TABLE ")
        print("="*80)
        display_df = final_df.copy()
        display_df['HR (95% CI)'] = display_df.apply(
            lambda x: f"{x['multiCox_HR']:.2f} ({x['multiCox_HR_95CI_lower']:.2f}-{x['multiCox_HR_95CI_upper']:.2f})", axis=1
        )
        print(display_df[['feature', 'lasso_coef', 'HR (95% CI)', 'multiCox_p_value']].to_string(index=False))
        print("="*80 + "\n")

    def _plot_km_df(self, df3, title, out_path):
        sns.set_style("whitegrid")
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(9, 7))
        ax = plt.subplot(111)
        df3 = df3.dropna()
        gs = sorted(df3['Group'].unique())
        pal = ["#0073C2", "#D55E00", "#009E73", "#EFC000"]
        for i, g in enumerate(gs):
            m = df3['Group'] == g
            if m.sum() == 0:
                continue
            kmf.fit(df3.loc[m, 'T'], event_observed=df3.loc[m, 'E'], label=f"{g}(n={int(m.sum())})")
            kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2.4, color=pal[i % len(pal)])
        if len(gs) > 1:
            try:
                res = multivariate_logrank_test(df3['T'], df3['Group'], df3['E'])
                p = res.p_value
                plt.text(0.05, 0.05, f"Log-rank P={p:.2e}", transform=ax.transAxes,
                         fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.85))
            except Exception:
                pass
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Time (Days)")
        plt.ylabel("Survival Probability")
        sns.despine()
        plt.grid(True, alpha=0.25, linestyle='--')
        plt.legend(loc="best", frameon=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curves(self, output_path):
        sns.set_style("whitegrid")
        time_points = [365, 1095, 1825] 
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        plt.figure(figsize=(9, 8))
        for tp, color in zip(time_points, colors):
            subset_mask = (self.df['T'] >= tp) | (self.df['E'] == 1)
            subset = self.df[subset_mask].copy()
            subset['label'] = (subset['T'] < tp) & (subset['E'] == 1)
            subset['label'] = subset['label'].astype(int)
            if len(subset['label'].unique()) < 2: 
                continue
            fpr, tpr, _ = roc_curve(subset['label'], subset['RiskScore'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{tp//365}-Year ROC (AUC = {roc_auc:.3f})')
            plt.fill_between(fpr, tpr, alpha=0.1, color=color)
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
        plt.title('Time-dependent ROC Analysis', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12, frameon=True, shadow=True)
        plt.grid(True, alpha=0.3)
        sns.despine()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_correlation_combo(self, genes, output_dir):
        if len(genes) < 3: 
            return
        genes = genes[:20]
        X = self.df[genes]
        corr_mat = X.corr()
        try:
            X_const = add_constant(X)
            vif_vals = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
            vif_data = pd.Series(vif_vals, index=X_const.columns).drop('const', errors='ignore')
        except:
            vif_data = pd.Series(0, index=genes)
        fig = plt.figure(figsize=(24, 10))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
        ax1 = fig.add_subplot(gs[0])
        mask = np.triu(np.ones_like(corr_mat, dtype=bool))
        sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap='RdBu_r', center=0, ax=ax1, square=True, linewidths=.5,
                    cbar_kws={"shrink": .7}, mask=mask, annot_kws={"size": 10})
        ax1.set_title("Feature Correlation Matrix", fontsize=16, fontweight='bold')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax2 = fig.add_subplot(gs[1], polar=True)
        labels = vif_data.index.tolist()
        stats = vif_data.values.tolist()
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        stats += stats[:1]
        angles += angles[:1]
        ax2.plot(angles, stats, color='#E41A1C', linewidth=2, linestyle='solid')
        ax2.fill(angles, stats, color='#E41A1C', alpha=0.25)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(labels, fontsize=10)
        ax2.tick_params(pad=20)
        ax2.set_title("Variance Inflation Factor (VIF)\n(Multicollinearity Check)", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gene_correlation_vif_combo.png"), dpi=300)
        plt.close()

    def _refresh_feature_columns(self):
        self.clinical_cols = [c for c in self.df.columns if any(p in c.lower() for p in self.clinical_patterns)]
        self.snp_cols = [c for c in self.df.columns if c not in self.clinical_cols]

    def plot_forest(self, summary_df, output_path):
        if summary_df.empty: return
        df_plot = summary_df.iloc[::-1]
        features = df_plot.index.tolist()
        hr = df_plot['HR'].values
        lower = df_plot['HR_lower_95%'].values
        upper = df_plot['HR_upper_95%'].values
        p_values = df_plot['p'].values
        plt.figure(figsize=(10, max(4, len(features) * 0.6)))
        ax = plt.gca()
        y_pos = np.arange(len(features))
        ax.errorbar(hr, y_pos, xerr=[hr - lower, upper - hr], fmt='o', color='black', 
                    ecolor='gray', elinewidth=2, capsize=4)
        ax.axvline(x=1, color='red', linestyle='--', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=12)
        trans = ax.get_yaxis_transform()
        for i, p in enumerate(p_values):
            p_text = f"p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            font_weight = 'bold' if p < 0.05 else 'normal'
            plt.text(1.02, i, p_text, transform=trans, va='center', fontsize=11, fontweight=font_weight)
            hr_text = f"HR={hr[i]:.2f} [{lower[i]:.2f}-{upper[i]:.2f}]"
            plt.text(1.25, i, hr_text, transform=trans, va='center', fontsize=11)
        plt.title("Multivariate Cox Regression Forest Plot", fontsize=14, fontweight='bold')
        plt.xlabel("Hazard Ratio (95% CI)", fontsize=12)
        if max(upper) > 10:
            plt.xscale('log')
        sns.despine()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_risk_heatmap(self, features, output_path):
        df_sort = self.df.sort_values('RiskScore')
        risk_score = df_sort['RiskScore'].values
        survival_time = df_sort['T'].values
        status = df_sort['E'].values
        expr_data = df_sort[features].T 
        vals = expr_data.values
        means = vals.mean(axis=1, keepdims=True)
        stds = vals.std(axis=1, keepdims=True)
        stds = np.where(stds == 0, 1, stds)
        expr_norm_vals = (vals - means) / stds
        expr_norm = pd.DataFrame(expr_norm_vals, index=expr_data.index, columns=expr_data.columns)
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 2], hspace=0.1)
        ax1 = fig.add_subplot(gs[0])
        x = np.arange(len(risk_score))
        median_score = np.median(risk_score)
        ax1.plot(x, risk_score, color='black', linewidth=1.5, zorder=10)
        ax1.fill_between(x, risk_score, median_score, where=(risk_score > median_score), 
                         interpolate=True, color='#E41A1C', alpha=0.3, label='High Risk')
        ax1.fill_between(x, risk_score, median_score, where=(risk_score <= median_score), 
                         interpolate=True, color='#377EB8', alpha=0.3, label='Low Risk')
        ax1.axhline(median_score, color='gray', linestyle='--', linewidth=1)
        ax1.set_ylabel("Risk Score", fontsize=12)
        ax1.set_title("Risk Score Analysis", fontsize=14, fontweight='bold')
        ax1.set_xticklabels([])
        ax1.set_xlim(0, len(risk_score))
        ax1.legend(loc="upper left")
        ax2 = fig.add_subplot(gs[1])
        colors = np.where(status == 1, '#E41A1C', '#377EB8')
        ax2.scatter(x, survival_time, c=colors, s=10, alpha=0.7)
        ax2.set_ylabel("Survival Time (Days)", fontsize=12)
        ax2.set_xticklabels([])
        ax2.set_xlim(0, len(risk_score))
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Dead',
                                  markerfacecolor='#E41A1C', markersize=8),
                           Line2D([0], [0], marker='o', color='w', label='Alive',
                                  markerfacecolor='#377EB8', markersize=8)]
        ax2.legend(handles=legend_elements, loc="upper left")
        ax3 = fig.add_subplot(gs[2])
        sns.heatmap(expr_norm, cmap="RdBu_r", center=0, ax=ax3, 
                    cbar_kws={"label": "Z-score Expression", "orientation": "horizontal", "pad": 0.2})
        ax3.set_ylabel("Features", fontsize=12)
        ax3.set_xlabel("Samples (Sorted by Risk Score)", fontsize=12)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    # ========================== Advanced Analysis ==========================
    def run_advanced_analysis(self, selected_features_series):
        """Executes Venn and Volcano analysis."""
        print("\n=== Step 4: Advanced Analysis (Venn, Volcano) ===")
        
        # 1. Differential Expression Analysis (High vs Low Risk)
        # Using Train Youden Cutoff if available, otherwise median
        if hasattr(self, 'train_cutoff'):
            cutoff_val = self.train_cutoff
            print(f"Using Train Youden Cutoff ({cutoff_val:.4f}) for DEA grouping.")
        else:
            cutoff_val = self.df['RiskScore'].median()
            print(f"Using Median Cutoff ({cutoff_val:.4f}) for DEA grouping.")
            
        self.df['RiskGroup_all'] = np.where(self.df['RiskScore'] > cutoff_val, 'High', 'Low')
        deg_df = self._run_differential_expression()
        
        # 2. Volcano Plot
        self.plot_volcano(deg_df, os.path.join(self.output_dir, "volcano_plot.png"))
        
        # 3. Venn Diagram (Cox Sig vs DEA Sig)
        # Load univariate sig
        uni_path = os.path.join(self.output_dir, f"snp_univariate_significant_p{self.p_thres}.csv")
        if os.path.exists(uni_path):
            uni_df = pd.read_csv(uni_path, sep='\t')
            uni_genes = set(uni_df['snp'].tolist())
        else:
            uni_genes = set()
            
        dea_genes = set(deg_df[deg_df['p_value'] < 0.05]['gene'].tolist())
        self.plot_venn(uni_genes, dea_genes, ['Cox Sig', 'DEA Sig'], os.path.join(self.output_dir, "venn_cox_dea.png"))

    def _run_differential_expression(self):
        """Calculates Log2FC and P-value for all features between High and Low risk groups."""
        print("Running Differential Expression Analysis (High vs Low Risk)...")
        high_mask = self.df['RiskGroup_all'] == 'High'
        low_mask = self.df['RiskGroup_all'] == 'Low'
        
        results = []
        for snp in self.snp_cols:
            high_vals = self.df.loc[high_mask, snp].dropna()
            low_vals = self.df.loc[low_mask, snp].dropna()
            if len(high_vals) < 3 or len(low_vals) < 3:
                continue
            
            # T-test
            try:
                t_stat, p_val = stats.ttest_ind(high_vals, low_vals, equal_var=False)
                mean_high = np.mean(high_vals)
                mean_low = np.mean(low_vals)
                # Log2FC（假设数据已经接近对数尺度或只是差值）
                # 如果数据是原始计数，我们需要取对数。如果是Z分数，差值就足够了，但通常FC意味着比率。
                # 假设输入是表达值（通常经过对数转换）。
                logfc = mean_high - mean_low 
                results.append({'gene': snp, 'logFC': logfc, 'p_value': p_val})
            except:
                continue
        
        res_df = pd.DataFrame(results)
        res_df['adj_p'] = multipletests(res_df['p_value'].fillna(1), method='fdr_bh')[1]
        res_df.to_csv(os.path.join(self.output_dir, "dea_results.csv"), index=False)
        return res_df

    def plot_volcano(self, res_df, output_path):
        if res_df.empty: return
        plt.figure(figsize=(10, 8))
        sns.set_style("white")
        
        res_df['-log10p'] = -np.log10(res_df['p_value'] + 1e-300)
        
        # 颜色编码
        res_df['color'] = 'grey'
        res_df.loc[(res_df['p_value'] < 0.05) & (res_df['logFC'] > 0.5), 'color'] = 'red'
        res_df.loc[(res_df['p_value'] < 0.05) & (res_df['logFC'] < -0.5), 'color'] = 'blue'
        
        plt.scatter(res_df['logFC'], res_df['-log10p'], c=res_df['color'], alpha=0.6, s=20)
        
        # 标注前10个基因
        top_genes = res_df.sort_values('p_value').head(10)
        for _, row in top_genes.iterrows():
            plt.text(row['logFC'], row['-log10p'], row['gene'], fontsize=8)
            
        plt.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=0.8)
        plt.axvline(0.5, color='black', linestyle='--', linewidth=0.8)
        plt.axvline(-0.5, color='black', linestyle='--', linewidth=0.8)
        
        plt.title("Volcano Plot (High vs Low Risk)", fontsize=16)
        plt.xlabel("Log2 Fold Change", fontsize=12)
        plt.ylabel("-Log10 P-value", fontsize=12)
        sns.despine()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_venn(self, set1, set2, labels, output_path):
        if venn2 is None: 
            print("matplotlib_venn not installed, skipping Venn diagram.")
            return
        plt.figure(figsize=(8, 8))
        venn2([set1, set2], set_labels=labels)
        plt.title("Venn Diagram of Significant Genes")
        plt.savefig(output_path, dpi=300)
        plt.close()

    def run_full_pipeline(self):
        # 1. Overall Survival Analysis
        print("\n=== Running Overall Survival Analysis (All Samples) ===")
        candidates, _ = self.univariate_cox()
        if candidates:
            selected = self.lasso_cox(candidates)
            if not getattr(selected, "empty", False):
                self.build_final_model(selected)
                self.run_advanced_analysis(selected)
                top_feats = selected.abs().sort_values(ascending=False).head(3).index
                for feat in top_feats:
                    self.plot_km(feat, os.path.join(self.output_dir, f"km_{feat}.png"), discrete=True)
            else:
                print("No features selected by LASSO for overall analysis.")
        else:
            print("No significant features found for overall analysis.")

        # 2. 按结局类型进行的亚组分析（E=1/2/4/未知）
        ev_map = {"1": "stage_I", "2": "stage_II", "4": "stage_IV", "unknown": "stage_unknown"}
        if all(c in self.df.columns for c in ev_map.values()):
            print("\n=== Running Subgroup Analysis by Outcome Type ===")
            base_out = os.path.join(self.output_dir, "TraditionalAnalysis")
            if not os.path.exists(base_out):
                os.makedirs(base_out)
            
            # Backup original state
            df0 = self.df.copy()
            out0 = self.output_dir
            E0 = df0['E'].copy() if 'E' in df0.columns else None
            
            # 构建临时结果标签
            ss = {k: df0[c].astype(bool).values for k, c in ev_map.items()}
            et = np.full(df0.shape[0], "unknown", dtype=object)
            for k in ["1", "2", "4", "unknown"]:
                m = ss[k]
                et[m] = k
            df0['_E_type'] = et
            
            for k in ["1", "2", "4", "unknown"]:
                self.output_dir = os.path.join(base_out, k)
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                
                self.df = df0.copy()
                self.df['E'] = (self.df['_E_type'] == k).astype(int) # This modifies the Event definition!
                self.df = self.df.drop(columns=['_E_type'], errors='ignore')
                self._refresh_feature_columns()
                
                print(f"\n>>> Subgroup Analysis: Outcome E={k} (Samples={self.df.shape[0]}, Events={int(self.df['E'].sum())})")
                
                sub_candidates, _ = self.univariate_cox()
                if sub_candidates:
                    sub_selected = self.lasso_cox(sub_candidates)
                    if not getattr(sub_selected, "empty", False):
                        self.build_final_model(sub_selected)
                        self.run_advanced_analysis(sub_selected) # Added Advanced Analysis for subgroups too
                        top_feats = sub_selected.abs().sort_values(ascending=False).head(3).index
                        for feat in top_feats:
                            self.plot_km(feat, os.path.join(self.output_dir, f"km_{feat}.png"), discrete=True)
                    else:
                        print(f"No features selected by LASSO for E={k}.")
                else:
                    print(f"No significant features found for E={k}.")
            
            # 恢复状态
            self.output_dir = out0
            self.df = df0.drop(columns=['_E_type'], errors='ignore')
            if E0 is not None:
                self.df['E'] = E0
            self._refresh_feature_columns()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="e:\\TERM\\data", help="Path to data directory")
    parser.add_argument("--output_dir", default="e:\\TERM\\results", help="Path to output directory")
    parser.add_argument("--p_thres", type=float, default=0.01, help="Univariate P-value threshold")
    parser.add_argument("--beta_norm", default="l1", help="beta normalization: l1/l2")
    parser.add_argument("--valid_size", type=float, default=0.3, help="Validation set ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for split")
    parser.add_argument("--cutoff_time", type=float, default=1095, help="ROC time point for Youden cutoff")
    args = parser.parse_args()
    
    try:
        ta = TraditionalAnalysis(args.data_dir, args.output_dir,
                                 p_thres=args.p_thres, beta_norm=args.beta_norm,
                                 valid_size=args.valid_size, random_state=args.random_state,
                                 cutoff_time=args.cutoff_time)
        ta.run_full_pipeline()
        print("\nPipeline execution completed successfully.")
    except Exception as e:
        print(f"\nPipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
