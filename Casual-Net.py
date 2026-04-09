import sys
import os
import numpy as np
import pandas as pd
import warnings
import logging

if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())
sys.path.append(os.getcwd())

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Dependency Checks & Imports ---
MISSING_PACKAGES = []

# Check CausalML Meta Learners (XLearner, DRLearner)
HAS_META = False
try:
    from causalml.inference.meta import BaseXRegressor, XGBDRRegressor
    from xgboost import XGBRegressor
    HAS_META = True
except ImportError as e:
    logger.warning(f"Meta Learners unavailable: {e}")
    MISSING_PACKAGES.append("xgboost")

# --- Data Loading ---
def load_data():
    data_path = r"e:\TERM\data\merged_dataset_normalized.tsv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return None, None, None, None

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, sep="\t", index_col=0)
    
    # Identify columns
    clinical_cols = [c for c in df.columns if c.startswith("age") or c.startswith("gender") or c.startswith("stage")]
    gene_cols = [c for c in df.columns if c.startswith("ENSG")]
    outcome_col = "T" if "T" in df.columns else df.columns[-1]
    
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Clinical covariates: {len(clinical_cols)}")
    logger.info(f"Gene candidates: {len(gene_cols)}")
    
    return df, clinical_cols, gene_cols, outcome_col

# --- Analysis Logic ---
def run_analysis():
    df, clinical_cols, gene_cols, outcome_col = load_data()
    if df is None:
        return

    # Load genes from single-factor Cox regression results
    cox_genes_path = r"e:\TERM\results\cox\snp_univariate_significant_p0.01.csv"
    if os.path.exists(cox_genes_path):
        try:
            cox_df = pd.read_csv(cox_genes_path, sep="\t")
            if 'snp' in cox_df.columns:
                target_genes = cox_df['snp'].tolist()
            else:
                target_genes = cox_df.iloc[:, 0].tolist()
            
            # Filter to ensure genes exist in the current dataset
            top_genes = [g for g in target_genes if g in gene_cols]
            logger.info(f"Loaded {len(top_genes)} valid genes from {cox_genes_path}")
            
            if not top_genes:
                logger.warning("No valid genes found in the Cox file that exist in the dataset. Exiting.")
                return
        except Exception as e:
            logger.error(f"Error loading Cox genes: {e}")
            return
    else:
        logger.error(f"Cox file not found: {cox_genes_path}")
        return

    results = []

    # Prepare Covariates (X) and Outcome (y)
    # --- MODIFICATION: Handle Survival Data (Binary Outcome) ---
    # Standard ATE models cannot handle censored data directly.
    # We convert to Binary Outcome: "Event within 3 Years (1095 days)"
    cutoff_days = 1095
    logger.info(f"Transforming Survival Data to Binary Outcome (Cutoff={cutoff_days} days)...")
    
    # Valid samples: (Survived > Cutoff) OR (Event <= Cutoff)
    # Exclude: (Censored < Cutoff) - outcome unknown
    mask_valid = (df[outcome_col] >= cutoff_days) | (df['E'] == 1)
    df_clean = df[mask_valid].copy()
    
    # Outcome: 1 = Event within 3 years (High Risk), 0 = Survived > 3 years (Low Risk)
    df_clean['Y_binary'] = ((df_clean[outcome_col] < cutoff_days) & (df_clean['E'] == 1)).astype(int)
    
    logger.info(f"Filtered samples for binary analysis: {len(df)} -> {len(df_clean)}")
    
    if df_clean.empty:
        logger.error("No valid samples left after filtering!")
        return

    X = df_clean[clinical_cols].values
    y = df_clean['Y_binary'].values

    for gene in top_genes:
        logger.info(f"Analyzing Gene/SNP: {gene}")
        
        # Treatment (t): Binarize gene expression (High vs Low)
        # Using median split
        t_continuous = df_clean[gene].values
        t_binary = (t_continuous > np.median(t_continuous)).astype(int)
        
        row = {'SNP_Site': gene}
        
        # --- Model 1: XLearner (Meta) ---
        if HAS_META:
            try:
                # BaseXRegressor with XGBoost
                xl = BaseXRegressor(learner=XGBRegressor(n_estimators=100, max_depth=3, n_jobs=1))
                xl.fit(X, t_binary, y)
                # predict(X) usually returns CATE array for meta learners
                cate_xl = xl.predict(X)
                row['XLearner'] = np.mean(cate_xl)
            except Exception as e:
                logger.error(f"XLearner failed for {gene}: {e}")
                row['XLearner'] = np.nan
        else:
            row['XLearner'] = np.nan

        # --- Model 2: DRLearner (Meta) ---
        if HAS_META:
            try:
                dr = XGBDRRegressor()
                dr.fit(X, t_binary, y)
                cate_dr = dr.predict(X)
                row['DRLearner'] = np.mean(cate_dr)
            except Exception as e:
                logger.error(f"DRLearner failed for {gene}: {e}")
                row['DRLearner'] = np.nan
        else:
            row['DRLearner'] = np.nan

        results.append(row)

    # Compile Results
    results_df = pd.DataFrame(results)
    
    output_dir = r"e:\TERM\results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not results_df.empty:
        model_cols = [c for c in results_df.columns if c != 'SNP_Site']
        
        # 1. Save separate files for each model (ATE > 0.01)
        for model in model_cols:
            # Filter rows where this specific model abs(ATE) > 0.01
            model_mask = results_df[model].fillna(0).abs() > 0.01
            model_df = results_df[model_mask]
            
            model_out_path = os.path.join(output_dir, f"casual_net_{model}.csv")
            model_df.to_csv(model_out_path, index=False)
            logger.info(f"Saved {model} results to {model_out_path} ({len(model_df)} genes)")

        # 2. Save Union (Any model abs(ATE) > 0.01)
        # Only for the models in this file (XLearner, DRLearner)
        union_mask = results_df[model_cols].fillna(0).abs().apply(lambda x: x > 0.01).any(axis=1)
        union_df = results_df[union_mask]
        
        union_path = os.path.join(output_dir, "casual_net_union.csv")
        union_df.to_csv(union_path, index=False)
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE (Meta Learners)")
        print(f"Union Results saved to: {union_path}")
        print("Individual model results saved to same directory.")
        print("Union Preview:")
        print(union_df.head())
        print("="*50 + "\n")
    else:
        logger.warning("No results generated.")

    # --- Missing Package Report ---
    if MISSING_PACKAGES:
        print("\n" + "!"*50)
        print("MISSING DEPENDENCIES DETECTED")
        print("Please run the following commands in your terminal to fix:")
        for pkg in set(MISSING_PACKAGES):
            if pkg == "xgboost":
                print("  pip install xgboost")
        print("!"*50 + "\n")

if __name__ == "__main__":
    run_analysis()
