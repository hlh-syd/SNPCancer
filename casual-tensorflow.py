import sys
import os
import numpy as np
import pandas as pd
import warnings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging

# --- Dependency Checks & Imports ---
HAS_TF = False
try:
    import tensorflow as tf
    from causalml.inference.tf import DragonNet
    HAS_TF = True
    logger.info("Successfully imported DragonNet from causalml.inference.tf")
except ImportError as e:
    logger.error(f"DragonNet unavailable: {e}")
    HAS_TF = False

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
    
    return df, clinical_cols, gene_cols, outcome_col

# --- Analysis Logic ---
def run_analysis():
    if not HAS_TF:
        logger.error("TensorFlow/DragonNet not available. Please install tensorflow==2.10.0")
        return

    df, clinical_cols, gene_cols, outcome_col = load_data()
    if df is None:
        return

    # Load genes from single-factor Cox regression results
    cox_genes_path = r"e:\TERM\results\snp_univariate_reliable.csv"
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
        t_continuous = df_clean[gene].values
        t_binary = (t_continuous > np.median(t_continuous)).astype(int)
        
        row = {'SNP_Site': gene}
        
        # --- Model: DragonNet ---
        try:
            # DragonNet parameters
            dn = DragonNet(neurons_per_layer=50, verbose=False, epochs=50)
            # fit_predict returns the ATE
            dn_ate = dn.fit_predict(X, t_binary, y).mean()
            row['DragonNet'] = dn_ate
        except Exception as e:
            logger.error(f"DragonNet failed for {gene}: {e}")
            row['DragonNet'] = np.nan

        results.append(row)

    # Compile Results
    results_df = pd.DataFrame(results)
    
    output_dir = r"e:\TERM\results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not results_df.empty:
        # Filter abs(ATE) > 0.01
        mask = results_df['DragonNet'].fillna(0).abs() > 0.01
        filtered_df = results_df[mask]
        
        output_path = os.path.join(output_dir, "casual_tensorflow_DragonNet.csv")
        filtered_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved DragonNet results to {output_path} ({len(filtered_df)} genes)")
        print(filtered_df.head())
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    run_analysis()
