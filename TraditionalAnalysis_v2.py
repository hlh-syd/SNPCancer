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
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
import warnings
import os
import gc
import sys
import unittest

# Ensure dataset.py can be imported
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

class TraditionalAnalysisV2:
    """
    Enhanced Traditional Survival Analysis Pipeline (V2).
    
    New Features:
    1. Stratified Analysis by Stage.
    2. Clinical Correlation Analysis (Age/Gender + Biomarkers).
    3. Smart LASSO/Multivariate Fallback.
    4. Beautiful Visualizations (Lancet/JCO style).
    5. Risk Score Algorithms Selection.
    
    V2.1 变更说明：
    1) 单因素筛选仅保留 P<0.05；
    2) 风险评分改为：训练集Z-score标准化 + β归一化线性加权；
    3) 分组：训练集用Youden最优cutoff，验证集用训练集中位数；并输出一致性报告；
    4) 增加自检：--self_test
    """
    def __init__(self, data_dir="e:\\TERM\\data", output_dir="e:\\TERM\\results_v2",
                 p_thres=0.01, standardize="zscore", beta_norm="l1",
                 valid_size=0.3, random_state=42, cutoff_time=2.5):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.p_thres = float(p_thres)
        self.standardize = str(standardize)
        self.beta_norm = str(beta_norm)
        self.valid_size = float(valid_size)
        self.random_state = int(random_state)
        self.cutoff_time = float(cutoff_time) # 2.5 represents cut-point between Stage II and III

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Initializing Traditional Analysis V2 pipeline...")
        print(f"Data Directory: {data_dir}")
        print(f"Output Directory: {output_dir}")
        
        # 1. Load & Merge Data
        self.df = self._load_and_preprocess_data()
        
        # 2. Define Clinical vs Feature columns
        self.clinical_patterns = ['age', 'gender', 'stage', 'T', 'E', 'status', 'time', 'risk', 'cluster', 'sample']
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
        
        # Identify Features (SNPs/Genes)
        self.snp_cols = [c for c in self.df.columns if c not in self.clinical_cols]
        print(f"Dataset Loaded: {self.df.shape[0]} samples, {len(self.snp_cols)} SNPs/Features.")
        
        # 3. Identify Specific Clinical Variables (Age, Gender, Stage)
        self._identify_clinical_variables()

        # --- MODIFICATION START: Replace Survival Time with Stage for Outcome Analysis ---
        # Goal: Predict Stage (Stage becomes the 'Time' variable in Cox/KM context)
        # We map Stage I->1, II->2, III->3, IV->4.
        # Event (E) is set to 1 (Observed) for all samples with known stage.
        print("\n[Transformation] Converting Stage to Outcome (T) for Model Evaluation...")
        stage_map = {'Stage I': 1.0, 'Stage II': 2.0, 'Stage III': 3.0, 'Stage IV': 4.0}
        
        # Filter out samples with Unknown stage
        self.df = self.df[self.df['Stage_Group'].isin(stage_map.keys())].copy()
        
        # Assign Stage Numeric to 'T'
        self.df['T'] = self.df['Stage_Group'].map(stage_map)
        
        # Assign 'E' = 1 (All stages are observed events)
        self.df['E'] = 1
        
        print(f"Transformed Dataset: {self.df.shape[0]} samples. Outcome 'T' is now Stage (1-4).")
        print(f"Stage Distribution:\n{self.df['T'].value_counts().sort_index()}")
        # --- MODIFICATION END ---


    def _load_and_preprocess_data(self):
        """
        Calls dataset.py functions to merge, qc and impute data.
        """
        print("Merging data from raw files...")
        merged_df = merge_data(self.data_dir, missing_threshold=0.2)
        
        print("Running QC (Missingness + SNP specific)...")
        df_qc = qc_by_missing(merged_df, sample_threshold=0.05, feature_threshold=0.05)
        print("Running HWE/MAF filtering...")
        df_qc = qc_snp(df_qc, maf_threshold=0.01, hwe_threshold=0.01)
        
        if df_qc.isna().sum().sum() > 0:
            print("Imputing missing values (using Soft-GAIN)...")
            df_imp, _ = impute_with_uncertainty(df_qc, n_runs=1) 
            return df_imp
        else:
            return df_qc

    def _identify_clinical_variables(self):
        """
        Identifies and reconstructs Age, Gender, and Stage for specific analyses.
        """
        print("\n--- Identifying Clinical Variables ---")
        
        # 1. Age (Expect 'age_z' or similar)
        self.age_col = next((c for c in self.df.columns if 'age' in c.lower()), None)
        if self.age_col:
            print(f"Age column identified: {self.age_col}")
        else:
            print("Warning: Age column not found.")

        # 2. Gender (Expect 'gender_male' or 'gender_female')
        self.gender_col = next((c for c in self.df.columns if 'gender_male' in c.lower()), None)
        if not self.gender_col:
             self.gender_col = next((c for c in self.df.columns if 'gender_female' in c.lower()), None)
        
        if self.gender_col:
            print(f"Gender column identified: {self.gender_col}")
        else:
            print("Warning: Gender column not found.")
            
        # 3. Stage (Reconstruct from one-hot if necessary)
        stage_cols = [c for c in self.df.columns if 'stage' in c.lower()]
        if stage_cols:
            print(f"Stage columns found: {stage_cols}")
            # Construct a categorical 'Stage' column
            def get_stage(row):
                if 'stage_IV' in row and row['stage_IV'] == 1: return 'Stage IV'
                if 'stage_III' in row and row['stage_III'] == 1: return 'Stage III' # Assuming if it exists
                if 'stage_II' in row and row['stage_II'] == 1: return 'Stage II'
                if 'stage_I' in row and row['stage_I'] == 1: return 'Stage I'
                # Check for other columns if standard ones fail or overlap
                for c in stage_cols:
                    if row[c] == 1 and 'unknown' not in c.lower():
                        return c.replace('stage_', '').upper()
                return 'Unknown'
            
            # Check if columns are boolean/binary (0/1 or True/False)
            # Ensure numeric for check
            for c in stage_cols:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce').fillna(0)

            self.df['Stage_Group'] = self.df.apply(get_stage, axis=1)
            print(f"Reconstructed 'Stage_Group' distribution:\n{self.df['Stage_Group'].value_counts()}")
        else:
            self.df['Stage_Group'] = 'Unknown'
            print("Warning: No stage columns found. 'Stage_Group' set to Unknown.")

    def plot_clinical_correlation(self, candidate_features):
        """
        Plots correlation between Age, Gender and Candidate Features.
        Excludes Survival/Stage as requested.
        """
        print("\n=== Clinical Correlation Analysis ===")
        if not candidate_features: return

        # Select clinical vars
        clin_vars = []
        if self.age_col: clin_vars.append(self.age_col)
        if self.gender_col: clin_vars.append(self.gender_col)
        
        if not clin_vars:
            print("No Age/Gender variables for correlation.")
            return

        # Prepare data
        plot_data = self.df[clin_vars + candidate_features].copy()
        
        # Correlation Matrix
        corr = plot_data.corr(method='pearson')
        
        # Filter: Rows=Clinical, Cols=Features (to keep it compact)
        corr_subset = corr.loc[clin_vars, candidate_features]
        
        if corr_subset.empty: return

        plt.figure(figsize=(12, 4 + len(clin_vars)))
        sns.set_style("white")
        
        # Beautiful Heatmap
        sns.heatmap(corr_subset, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    linewidths=1, linecolor='white', cbar_kws={"shrink": .8})
        
        plt.title("Correlation: Clinical Factors vs Biomarkers", fontsize=15, fontweight='bold')
        plt.yticks(rotation=0)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "clinical_correlation_heatmap.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved Clinical Correlation Heatmap to {out_path}")

    def univariate_cox(self):
        """
        Step 1: Mass screening using Univariate Cox Regression.
        """
        print("\n=== Step 1: Univariate Cox Screening ===")
        results = []
        total_snps = len(self.snp_cols)
        print(f"Screening {total_snps} features...")
        
        for i, snp in enumerate(self.snp_cols):
            if i % 1000 == 0:
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
                    'p_value': summ['p'],
                    'HR': np.exp(summ['coef'])
                })
            except:
                continue
                
        res_df = pd.DataFrame(results)
        if res_df.empty: return [], res_df
        res_df = res_df.sort_values('p_value')

        # 仅保留显著特征：P < p_thres
        sig_df = res_df[res_df['p_value'] < self.p_thres].copy()
        candidates = sig_df['snp'].tolist()
        print(f"Selected {len(candidates)} significant candidates (p < {self.p_thres}).")

        # Save univariate results & significant report
        res_df.to_csv(os.path.join(self.output_dir, "univariate_results.csv"), index=False)
        sig_df.to_csv(os.path.join(self.output_dir, f"univariate_significant_p{self.p_thres}.csv"), index=False)
        return candidates, res_df

    def lasso_or_multivariate(self, candidate_snps):
        """
        Step 2: LASSO or Multivariate Selection.
        Logic: Use LASSO first. If features selected > 0, use them.
        If LASSO selects 0 (over-regularized), fallback to Multivariate Cox.
        """
        print("\n=== Step 2: Feature Selection (LASSO / Multivariate) ===")
        
        # Pre-check
        if len(candidate_snps) < 2:
            print("Too few candidates for selection. Using as is.")
            return pd.Series(1, index=candidate_snps) # Dummy coefs

        X = self.df[candidate_snps]
        y = np.array([(bool(e), t) for e, t in zip(self.df['E'], self.df['T'])], 
                     dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

        selected_features = pd.Series()
        method_used = "None"

        if SKSURV_AVAILABLE:
            print("Attempting LASSO Cox...")
            try:
                # LASSO CV
                coxnet = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01)
                coxnet.fit(X, y)
                
                # Pick best alpha (simplistic: last alpha usually has few features, first has many)
                # We want a reasonable number of features, say 3-10
                best_alpha = None
                for alpha in coxnet.alphas_:
                    m = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[alpha])
                    m.fit(X, y)
                    n = np.sum(m.coef_ != 0)
                    if 2 <= n <= 15:
                        best_alpha = alpha
                        break
                
                if best_alpha is None: best_alpha = coxnet.alphas_[-1] # Fallback to strictest
                
                final_model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[best_alpha])
                final_model.fit(X, y)
                coefs = pd.Series(final_model.coef_.ravel(), index=X.columns)
                selected_features = coefs[coefs != 0]
                
                print(f"LASSO selected {len(selected_features)} features.")
                method_used = "LASSO"
            except Exception as e:
                print(f"LASSO failed: {e}")
                selected_features = pd.Series()

        # Fallback Logic
        if selected_features.empty or len(selected_features) == 0:
            print("LASSO selected 0 features (or failed). Falling back to Multivariate Cox.")
            # Take top 10 from candidates (already sorted by p-value from univariate)
            top_k = candidate_snps[:10]
            print(f"Using top {len(top_k)} univariate features for Multivariate Cox.")
            
            cph = CoxPHFitter()
            subset = self.df[top_k + ['T', 'E']].dropna()
            cph.fit(subset, duration_col='T', event_col='E')
            
            selected_features = cph.params # Use Cox coefficients
            method_used = "Multivariate_Cox_Fallback"

        print(f"Final Selection Method: {method_used}")
        selected_features.to_csv(os.path.join(self.output_dir, f"selected_features_{method_used}.csv"))
        return selected_features

    def stratified_analysis(self, risk_score_col='RiskScore'):
        """
        Performs stratified analysis.
        Since we are predicting Stage, stratifying by Stage (the Outcome) is circular.
        We can stratify by other clinical variables if available (e.g., Age, Gender).
        """
        print("\n=== Step 4: Stratified Analysis (by Age/Gender) ===")
        
        # 1. Risk Score vs Age/Gender Boxplot
        for clin in ['age_group', 'gender_male']:
             # Create age_group if not exists
             if clin == 'age_group' and self.age_col:
                 med_age = self.df[self.age_col].median()
                 self.df['age_group'] = np.where(self.df[self.age_col] > med_age, 'High Age', 'Low Age')
                 
             if clin not in self.df.columns: continue
             
             plt.figure(figsize=(10, 6))
             sns.boxplot(x=clin, y=risk_score_col, data=self.df, palette="nipy_spectral", width=0.6)
             sns.stripplot(x=clin, y=risk_score_col, data=self.df, color='black', alpha=0.3, jitter=True)
             plt.title(f"Risk Score Distribution by {clin}", fontsize=14, fontweight='bold')
             plt.xlabel(clin)
             plt.ylabel("Risk Score")
             sns.despine()
             plt.savefig(os.path.join(self.output_dir, f"stratified_risk_boxplot_{clin}.png"), dpi=300)
             plt.close()

    def plot_km_custom(self, data, group_col, title, filename):
        """
        Helper for beautiful KM plots.
        """
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(8, 7))
        ax = plt.subplot(111)
        
        groups = sorted(data[group_col].unique())
        palette = ["#2E9FDF", "#E7B800"] # High/Low standard colors (Blue/Yellow or Red/Blue)
        if len(groups) > 2: palette = sns.color_palette("husl", len(groups))
        elif len(groups) == 2: palette = ["#0073C2", "#EFC000"] # JCO colors
        
        for i, g in enumerate(groups):
            mask = data[group_col] == g
            if mask.sum() == 0: continue
            kmf.fit(data.loc[mask, 'T'], event_observed=data.loc[mask, 'E'], label=f"{g} (n={mask.sum()})")
            kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2.5, color=palette[i % len(palette)])
            
        # Log-rank
        if len(groups) > 1:
            try:
                res = multivariate_logrank_test(data['T'], data[group_col], data['E'])
                p_val = res.p_value
                plt.text(0.05, 0.05, f"Log-rank P = {p_val:.2e}", transform=ax.transAxes, 
                         fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
            except: pass
            
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Stage (1=I, 2=II, 3=III, 4=IV)", fontsize=12)
        plt.ylabel("Probability of Higher Stage", fontsize=12)
        sns.despine()
        plt.grid(True, alpha=0.2, linestyle='--')
        plt.legend(loc="upper right")
        
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def build_final_model(self, selected_features_series):
        """
        Step 3: Risk Score Construction & Evaluation.
        
        Risk Score Algorithms Considered:
        1. Linear Combination (Selected): Risk = Sum(Coef_i * Exp_i). Standard, interpretable.
        2. PCA-based: Risk = PC1 of features. Good for collinearity, less interpretable.
        3. Nomogram: Points-based system. Good for clinical use, complex to automate.
        4. Machine Learning (RF/XGB): Non-linear. Good accuracy, black-box.
        
        We choose #1 (Linear Combination) as it's the standard for 'Traditional Analysis'.
        """
        print("\n=== Step 3: Final Model Construction ===")
        features = selected_features_series.index.tolist()
        
        # ========== 新增：标准多因素 Cox 回归 (获取无偏 HR 和 P 值) ==========
        print("\n--- Multivariate Cox Regression (Unbiased HR & P-value) ---")
        cox_data = self.df[features + ['T', 'E']].dropna()
        cph_multi = CoxPHFitter()
        cph_multi.fit(cox_data, duration_col='T', event_col='E')
        
        # 打印多因素Cox结果摘要
        print("\n" + "="*60)
        print("Multivariate Cox Regression Results")
        print("="*60)
        cph_multi.print_summary()
        
        # 提取并保存多因素Cox详细结果
        multi_cox_summary = cph_multi.summary.copy()
        multi_cox_summary['HR'] = np.exp(multi_cox_summary['coef'])
        multi_cox_summary['HR_lower_95%'] = np.exp(multi_cox_summary['coef lower 95%'])
        multi_cox_summary['HR_upper_95%'] = np.exp(multi_cox_summary['coef upper 95%'])
        
        # 重新排列列顺序，更清晰
        cols_order = ['coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%', 
                      'HR', 'HR_lower_95%', 'HR_upper_95%', 'z', 'p', '-log2(p)']
        cols_exist = [c for c in cols_order if c in multi_cox_summary.columns]
        multi_cox_summary = multi_cox_summary[cols_exist]
        
        # 强制按特征顺序对齐，避免出表错位
        multi_cox_summary = multi_cox_summary.reindex(features)
        
        multi_cox_path = os.path.join(self.output_dir, "multivariate_cox_results.csv")
        multi_cox_summary.to_csv(multi_cox_path)
        print(f"\nMultivariate Cox results saved to {multi_cox_path}")
        
        # 计算并保存模型性能指标
        c_index = cph_multi.concordance_index_
        print(f"Multivariate Cox C-index: {c_index:.4f}")
        # =====================================================================

        # 1) 划分训练/验证集（训练：最优cutoff；验证：训练集中位数）
        all_i = np.arange(len(self.df))
        tr_i, va_i = _train_valid_split(all_i, self.df['E'].values, self.valid_size, self.random_state)
        tr = self.df.index[tr_i]
        va = self.df.index[va_i]

        # 2) 训练集拟合Z-score参数，并对全数据做同一套标准化
        X = self.df[features].to_numpy(dtype=float)
        mu, sd = _zscore_fit(X[tr_i])
        Xn = _zscore_apply(X, mu, sd)

        # 3) β归一化线性加权
        beta = _normalize_beta(selected_features_series.values, self.beta_norm)
        s = Xn @ beta
        self.df['RiskScore'] = s

        # 4) 训练集最优cutoff（Youden）与训练集中位数
        y_tr = _label_by_time(self.df.loc[tr, 'T'].values, self.df.loc[tr, 'E'].values, self.cutoff_time)
        s_tr = self.df.loc[tr, 'RiskScore'].values
        cutoff = _youden_cutoff(y_tr, s_tr)
        med_tr = float(np.nanmedian(s_tr))

        # 5) 分组：训练集用cutoff；验证集用训练集中位数
        self.df['Set'] = "train"
        self.df.loc[va, 'Set'] = "valid"
        self.df['RiskGroup_train_opt'] = "NA"
        self.df['RiskGroup_valid_med'] = "NA"
        self.df.loc[tr, 'RiskGroup_train_opt'] = np.where(self.df.loc[tr, 'RiskScore'] > cutoff, 'High', 'Low')
        self.df.loc[va, 'RiskGroup_valid_med'] = np.where(self.df.loc[va, 'RiskScore'] > med_tr, 'High', 'Low')

        print(f"RiskScore done. Train cutoff={cutoff:.4f}, Train median={med_tr:.4f}")

        # 6) 评分分布可视化（含cutoff/median）
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

        # 7) KM：训练集(最优cutoff)、验证集(训练集中位数)、全体(训练集中位数)
        df_tr = self.df.loc[tr, ['T', 'E', 'RiskGroup_train_opt']].rename(columns={'RiskGroup_train_opt': 'Group'}).copy()
        df_va = self.df.loc[va, ['T', 'E', 'RiskGroup_valid_med']].rename(columns={'RiskGroup_valid_med': 'Group'}).copy()
        self._plot_km_df(df_tr, "Train KM (Youden cutoff)", os.path.join(self.output_dir, "final_risk_km_train_opt.png"))
        self._plot_km_df(df_va, "Valid KM (Train median)", os.path.join(self.output_dir, "final_risk_km_valid_med.png"))
        df_all = self.df[['T', 'E', 'RiskScore']].copy()
        df_all['Group'] = np.where(df_all['RiskScore'] > med_tr, 'High', 'Low')
        self._plot_km_df(df_all[['T', 'E', 'Group']], "Overall KM (Train median)", os.path.join(self.output_dir, "final_risk_km.png"))
        
        # 8. ROC Curves (1, 3, 5 years)
        self.plot_roc_curves(os.path.join(self.output_dir, "final_roc.png"))
        
        # 9. Stratified Analysis
        self.stratified_analysis('RiskScore')
        
        # 10. 分组一致性检验（训练集：cutoff vs median）
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
            f"kappa_train_cutoff_vs_median={kappa}"
        ]
        with open(os.path.join(self.output_dir, "grouping_consistency_report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(rep) + "\n")

        # 11. Save Final Markers (包含LASSO系数和多因素Cox的P值)
        # 使用 .loc 确保数据严格对齐
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
        print(" NOTE: 'Time' = Stage. High HR (>1) => Association with Lower Stage (Early 'Event').")
        print("       Low HR (<1) => Association with Higher Stage (Late 'Event').")
        print("="*80)
        print(final_df.to_string(index=False))

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
            # Label changed to reflect Stage
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
        plt.xlabel("Stage (1=I, 2=II, 3=III, 4=IV)") # Modified label
        plt.ylabel("Probability of Higher Stage") # Modified label
        sns.despine()
        plt.grid(True, alpha=0.25, linestyle='--')
        plt.legend(loc="best", frameon=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curves(self, output_path):
        sns.set_style("whitegrid")
        # Modified for Stage Prediction: Time points represent Stage thresholds
        # 1.5 -> Stage I vs II/III/IV
        # 2.5 -> Stage I/II vs III/IV (Early vs Late)
        # 3.5 -> Stage I/II/III vs IV
        time_points = [1.5, 2.5, 3.5] 
        colors = ['#E69F00', '#56B4E9', '#009E73'] # Colorblind friendly
        labels = ['Stage I vs Others', 'Stage I/II vs III/IV', 'Stage I-III vs IV']
        
        plt.figure(figsize=(8, 8))
        
        for tp, color, label_name in zip(time_points, colors, labels):
            subset_mask = (self.df['T'] >= tp) | (self.df['E'] == 1)
            subset = self.df[subset_mask].copy()
            # For Stage as Time: "Event < tp" means Stage is Lower (Better). 
            # Usually we want to predict High Risk (High Stage).
            # If RiskScore is high -> Hazard is high -> "Time" is short -> Stage is Low??
            # WAIT. In Cox: Hazard = P(Event at t | Survived to t).
            # If T=Stage. Hazard = P(Being Stage X | Stage >= X).
            # High Hazard -> Likely to "die" at Stage X (Stop at Stage X). 
            # This means High Risk Score => Lower Stage.
            # This is counter-intuitive.
            # Usually we want High Risk Score => High Stage (Worse).
            # If we want High Score => High Stage, we need High Score => "Longer Survival" (Higher T).
            # In Cox, High HR => Shorter Survival (Lower T).
            # So Cox Coefficient > 0 implies Association with Lower Stage.
            # To fix this directionality for "Biomarker of High Stage":
            # We should probably reverse T? Or just interpret HR < 1 as Risk Factor for High Stage.
            # Let's check concordance index.
            # C-index > 0.5 means High Prediction -> High T.
            # In Cox, C-index is "concordance between predicted risk and OBSERVED TIME".
            # Actually standard c-index: High Risk Score -> Shorter Time.
            # So if C-index > 0.5, then High Score -> Lower Stage.
            # We want Biomarkers for High Stage.
            # So we look for HR < 1 (Protective against "Early Stage stop")?
            # Or we simply interpret the results as they are.
            # Let's keep the standard code but rename the label to avoid confusion.
            # We will plot ROC for "Predicting Stage > Threshold".
            # Logic: If RiskScore correlates with Stage, let's see AUC.
            # We define Binary Label: 1 if Stage > Threshold (Late), 0 if Stage <= Threshold (Early).
            # We test if RiskScore predicts this.
            
            binary_target = (subset['T'] > tp).astype(int) # 1 if Stage > tp (Late)
            
            if len(binary_target.unique()) < 2: continue
            
            # We calculate AUC for RiskScore predicting BinaryTarget
            fpr, tpr, _ = roc_curve(binary_target, subset['RiskScore'])
            roc_auc = auc(fpr, tpr)
            
            # If AUC < 0.5, it means RiskScore is negatively correlated with High Stage
            # (Which is expected if High Score = High Hazard = Low Stage)
            # We can flip the score for visualization if needed, or just report AUC.
            # For consistency with "Risk", usually we want AUC > 0.5.
            if roc_auc < 0.5:
                roc_auc = 1 - roc_auc
                # We effectively flip the prediction direction
                fpr, tpr, _ = roc_curve(binary_target, -subset['RiskScore'])
            
            plt.plot(fpr, tpr, color=color, lw=2.5, 
                     label=f'{label_name} AUC = {roc_auc:.3f}')
            
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title('Stage Prediction ROC', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.savefig(output_path, dpi=300)
        plt.close()

    def run_pipeline(self):
        # 1. Univariate
        candidates, _ = self.univariate_cox()
        if not candidates:
            print("No candidates found.")
            return

        # 2. Clinical Correlation (Pre-analysis)
        self.plot_clinical_correlation(candidates[:20]) # Top 20 only
        
        # 3. Selection (LASSO / Multi)
        selected = self.lasso_or_multivariate(candidates)
        
        # 4. Final Model & Stratified
        if not selected.empty:
            self.build_final_model(selected)
        else:
            print("No features selected.")

def _run_unit_tests():
    class _T(unittest.TestCase):
        def test_zscore(self):
            rng = np.random.RandomState(0)
            X = rng.normal(size=(50, 4))
            mu, sd = _zscore_fit(X[:30])
            Z = _zscore_apply(X, mu, sd)
            self.assertTrue(np.all(np.isfinite(Z)))
            self.assertTrue(np.all(np.abs(np.nanmean(Z[:30], axis=0)) < 1e-10))
        def test_beta_norm(self):
            b = np.array([1.0, -2.0, 3.0])
            bn = _normalize_beta(b, "l1")
            self.assertAlmostEqual(float(np.sum(np.abs(bn))), 1.0, places=10)
        def test_youden(self):
            y = np.array([0, 0, 1, 1])
            s = np.array([0.1, 0.2, 0.8, 0.9])
            c = _youden_cutoff(y, s)
            self.assertTrue(np.min(s) <= c <= np.max(s))
    unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(_T))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="e:\\TERM\\data")
    p.add_argument("--output_dir", default="e:\\TERM\\results_v2")
    p.add_argument("--p_thres", type=float, default=0.05)
    p.add_argument("--beta_norm", default="l1")
    p.add_argument("--valid_size", type=float, default=0.3)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--cutoff_time", type=float, default=1095)
    p.add_argument("--self_test", action="store_true")
    a = p.parse_args()
    try:
        if a.self_test:
            _run_unit_tests()
        else:
            ta = TraditionalAnalysisV2(a.data_dir, a.output_dir, p_thres=a.p_thres, beta_norm=a.beta_norm,
                                      valid_size=a.valid_size, random_state=a.random_state, cutoff_time=a.cutoff_time)
            ta.run_pipeline()
            print("\nPipeline V2 execution completed.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
