import sys
import os
import numpy as np
import pandas as pd
import logging
import warnings

# --- 1. Path Setup ---
# Add project root to sys.path to allow importing the local causalml source if needed
PROJECT_ROOT = r"e:\TERM"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 2. Model Import ---
HAS_FOREST = False
HAS_TREE = False

# Import CausalRandomForestRegressor
try:
    from causalml.inference.tree import CausalRandomForestRegressor
    HAS_FOREST = True
    logger.info("Successfully imported CausalRandomForestRegressor.")
except ImportError as e:
    logger.warning(f"Failed to import CausalRandomForestRegressor: {e}")
    # Try local source fallback
    try:
        from causalml.inference.tree.causal.causalforest import CausalRandomForestRegressor
        HAS_FOREST = True
        logger.info("Successfully imported CausalRandomForestRegressor from local source.")
    except ImportError as e2:
        logger.error(f"Failed to import CausalRandomForestRegressor from local source: {e2}")

# Import CausalTreeRegressor
try:
    from causalml.inference.tree import CausalTreeRegressor
    HAS_TREE = True
    logger.info("Successfully imported CausalTreeRegressor.")
except ImportError as e:
    logger.warning(f"Failed to import CausalTreeRegressor: {e}")
    # Try local source fallback
    try:
        from causalml.inference.tree.causal.causaltree import CausalTreeRegressor
        HAS_TREE = True
        logger.info("Successfully imported CausalTreeRegressor from local source.")
    except ImportError as e2:
        logger.error(f"Failed to import CausalTreeRegressor from local source: {e2}")

# --- 3. Data Loading ---
def load_data():
    data_path = os.path.join(PROJECT_ROOT, "data", "merged_dataset_normalized.tsv")
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return None, None, None, None

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, sep="\t", index_col=0)
    
    clinical_cols = [c for c in df.columns if c.startswith("age") or c.startswith("gender") or c.startswith("stage")]
    gene_cols = [c for c in df.columns if c.startswith("ENSG")]
    outcome_col = "T" if "T" in df.columns else df.columns[-1]
    
    logger.info(f"Data shape: {df.shape}")
    return df, clinical_cols, gene_cols, outcome_col

# --- 4. Main Analysis ---
def run_analysis():
    if not HAS_FOREST and not HAS_TREE:
        logger.error("No Tree/Forest models available. Exiting.")
        return

    df, clinical_cols, gene_cols, outcome_col = load_data()
    if df is None:
        return

    # Load genes from Cox results
    cox_genes_path = os.path.join(PROJECT_ROOT, "results", "cox", "snp_univariate_significant_p0.01.csv")
    if os.path.exists(cox_genes_path):
        try:
            cox_df = pd.read_csv(cox_genes_path, sep="\t")
            if 'snp' in cox_df.columns:
                target_genes = cox_df['snp'].tolist()
            else:
                target_genes = cox_df.iloc[:, 0].tolist()
            
            top_genes = [g for g in target_genes if g in gene_cols]
            logger.info(f"Loaded {len(top_genes)} valid genes from {cox_genes_path}")
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
        logger.info(f"Analyzing Gene: {gene}")
        
        # Treatment (t): Binarize gene expression (High vs Low)
        t_continuous = df_clean[gene].values
        t_binary = (t_continuous > np.median(t_continuous)).astype(int)
        
        row = {'SNP_Site': gene}
        
        # --- Model 1: Causal Forest ---
        if HAS_FOREST:
            try:
                crf = CausalRandomForestRegressor(n_estimators=50, n_jobs=1, min_samples_leaf=10)
                crf.fit(X, t_binary, y)
                
                if hasattr(crf, 'estimate_ate'):
                    ate_res = crf.estimate_ate(X, t_binary, y)
                    if isinstance(ate_res, (list, tuple, np.ndarray)):
                        ate_val = np.array(ate_res).flatten()[0]
                    else:
                        ate_val = ate_res
                else:
                    cate = crf.predict(X)
                    ate_val = np.mean(cate)
                    
                row['CausalForest'] = ate_val
            except Exception as e:
                logger.error(f"CausalForest failed for {gene}: {e}")
                row['CausalForest'] = np.nan
        else:
            row['CausalForest'] = np.nan

        # --- Model 2: Causal Tree ---
        if HAS_TREE:
            try:
                ct = CausalTreeRegressor()
                ct.fit(X, t_binary, y)
                
                if hasattr(ct, 'estimate_ate'):
                    ate_res = ct.estimate_ate(X, t_binary, y)
                    if isinstance(ate_res, (list, tuple, np.ndarray)):
                        ate_val = np.array(ate_res).flatten()[0]
                    else:
                        ate_val = ate_res
                else:
                    # Fallback if estimate_ate is missing (unlikely for CausalTree)
                    # But CausalTreeRegressor typically has it.
                    # Some versions might rely on predict -> mean
                    try:
                        cate = ct.predict(X)
                        ate_val = np.mean(cate)
                    except:
                         ate_val = np.nan

                row['CausalTree'] = ate_val
            except Exception as e:
                logger.error(f"CausalTree failed for {gene}: {e}")
                row['CausalTree'] = np.nan
        else:
             row['CausalTree'] = np.nan

        results.append(row)

    # --- 5. Save Results ---
    results_df = pd.DataFrame(results)
    
    output_dir = os.path.join(PROJECT_ROOT, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not results_df.empty:
        model_cols = [c for c in results_df.columns if c != 'SNP_Site']
        
        for model in model_cols:
            # Filter rows where this specific model abs(ATE) > 0.01
            mask = results_df[model].fillna(0).abs() > 0.01
            filtered_df = results_df[mask]
            
            if not filtered_df.empty:
                output_path = os.path.join(output_dir, f"casual_forest_{model}.csv")
                filtered_df[['SNP_Site', model]].to_csv(output_path, index=False)
                logger.info(f"Saved {model} results to {output_path} ({len(filtered_df)} genes)")
            else:
                logger.info(f"No significant results for {model} (abs(ATE) > 0.01)")
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    run_analysis()
