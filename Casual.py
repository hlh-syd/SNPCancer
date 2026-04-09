import sys
import os
import numpy as np
import pandas as pd
import logging
import warnings
from sklearn.model_selection import train_test_split

# Add project root to path to ensure imports work
sys.path.append(r"e:\TERM")

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("causalml").setLevel(logging.ERROR)
logging.getLogger("pyro").setLevel(logging.ERROR)

# Check for Pyro
try:
    import pyro
    HAS_PYRO = True
except ImportError:
    HAS_PYRO = False
    print("Warning: 'pyro' module not found. CEVAE will be skipped.")

# Check for XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: 'xgboost' not found. XGBoost-based models (XLearner, DRLearner) will be skipped.")

# Import CausalML models
HAS_CAUSALML_TORCH = False
HAS_CAUSALML_TF = False
HAS_CAUSALML_TREE = False
HAS_CAUSALML_META = False

# 1. CEVAE (Torch + Pyro)
if HAS_PYRO:
    try:
        from causalml.inference.torch.cevae import CEVAE
        HAS_CAUSALML_TORCH = True
    except ImportError as e:
        print(f"Warning: Could not import CEVAE: {e}")

# 2. DragonNet (TF)
try:
    from causalml.inference.tf.dragonnet import DragonNet
    HAS_CAUSALML_TF = True
except ImportError as e:
    print(f"Warning: Could not import DragonNet: {e}")

# 3. Causal Forest (Tree)
try:
    from causalml.inference.tree.causal.causalforest import CausalRandomForestRegressor
    HAS_CAUSALML_TREE = True
except ImportError as e:
    print(f"Warning: Could not import CausalRandomForestRegressor: {e}")

# 4. Meta Learners (XLearner, DRLearner)
try:
    from causalml.inference.meta.xlearner import BaseXRegressor
    from causalml.inference.meta.drlearner import XGBDRRegressor
    HAS_CAUSALML_META = True
except ImportError as e:
    print(f"Warning: Could not import MetaLearners (XLearner/DRLearner): {e}")

def load_data():
    """
    Load normalized dataset and prepare for causal analysis.
    """
    data_path = r"e:\TERM\data\merged_dataset_normalized.tsv"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path, sep="\t", index_col=0)
    
    # Identify Covariates (Clinical), Treatment candidates (Genes), and Outcome (T/E)
    # Based on inspection:
    # Clinical: age_z, gender_female, gender_male, stage_...
    # Outcome: T (Time)
    # Candidates: ENSG...
    
    clinical_cols = [c for c in df.columns if c.startswith("age") or c.startswith("gender") or c.startswith("stage")]
    outcome_col = "T" # Using normalized Time as outcome
    
    # Filter genes (assuming they start with ENSG)
    gene_cols = [c for c in df.columns if c.startswith("ENSG")]
    
    if outcome_col not in df.columns:
        # Fallback if T is not found (maybe not normalized?)
        # Try finding a column that looks like time or label
        outcome_col = df.columns[-2] # Guessing
    
    print(f"Loaded data: {df.shape}")
    print(f"Covariates: {len(clinical_cols)}")
    print(f"Genes (SNP sites candidates): {len(gene_cols)}")
    
    return df, clinical_cols, gene_cols, outcome_col

def run_causal_models(X, t, y):
    """
    Run 5 causal models and return ATEs.
    """
    results = {}
    
    # 1. XGBDRRegressor (DR-Learner)
    if HAS_XGBOOST and HAS_CAUSALML_META:
        try:
            dr = XGBDRRegressor()
            dr.fit(X, t, y)
            # estimate_ate returns (ate, lb, ub) usually, or we use predict -> mean
            cate_dr = dr.predict(X)
            results['DRLearner'] = np.mean(cate_dr)
        except Exception as e:
            results['DRLearner'] = np.nan
    else:
        results['DRLearner'] = np.nan

    # 2. BaseXRegressor (X-Learner)
    if HAS_XGBOOST and HAS_CAUSALML_META:
        try:
            xl = BaseXRegressor(learner=XGBRegressor(n_estimators=100, max_depth=3, n_jobs=1))
            xl.fit(X, t, y)
            cate_xl = xl.predict(X)
            results['XLearner'] = np.mean(cate_xl)
        except Exception as e:
            results['XLearner'] = np.nan
    else:
        results['XLearner'] = np.nan

    # 3. CausalRandomForestRegressor
    if HAS_CAUSALML_TREE:
        try:
            cf = CausalRandomForestRegressor(n_estimators=50, min_samples_leaf=10)
            cf.fit(X, t, y)
            ite_cf = cf.predict(X)
            results['CausalForest'] = np.mean(ite_cf)
        except Exception as e:
            results['CausalForest'] = np.nan
    else:
        results['CausalForest'] = np.nan

    # 4. DragonNet (TF)
    if HAS_CAUSALML_TF:
        try:
            dn = DragonNet(epochs=10, batch_size=64, verbose=False)
            dn.fit(X, t, y)
            ite_dn = dn.predict_tau(X)
            results['DragonNet'] = np.mean(ite_dn)
        except Exception as e:
            results['DragonNet'] = np.nan
    else:
        results['DragonNet'] = np.nan

    # 5. CEVAE (Torch)
    if HAS_CAUSALML_TORCH:
        try:
            ce = CEVAE(outcome_dist='normal', num_epochs=10, batch_size=64, latent_dim=10, hidden_dim=64)
            ce.fit(X, t, y)
            ite_ce = ce.predict(X)
            results['CEVAE'] = np.mean(ite_ce)
        except Exception as e:
            results['CEVAE'] = np.nan
    else:
        results['CEVAE'] = np.nan
        
    return results

def main():
    df, clinical_cols, gene_cols, outcome_col = load_data()
    
    # Load genes from single-factor Cox regression results
    cox_genes_path = r"e:\TERM\results\snp_univariate_reliable.csv"
    if os.path.exists(cox_genes_path):
        try:
            cox_df = pd.read_csv(cox_genes_path, sep="\t")
            if 'snp' in cox_df.columns:
                target_genes = cox_df['snp'].tolist()
            else:
                target_genes = cox_df.iloc[:, 0].tolist()
            
            top_genes = [g for g in target_genes if g in gene_cols]
            print(f"Loaded {len(top_genes)} valid genes from {cox_genes_path}")
            
            if not top_genes:
                print("No valid genes found in the Cox file that exist in the dataset. Exiting.")
                return
        except Exception as e:
            print(f"Error loading Cox genes: {e}")
            return
    else:
        print(f"Cox file not found: {cox_genes_path}")
        return
    
    # Prepare storage
    final_results = []
    
    # Pre-compute X (Covariates)
    # Using clinical features only to save compute
    X = df[clinical_cols].values
    y = df[outcome_col].values
    
    # Loop over each "SNP" (Gene) as Treatment
    for gene in top_genes:
        print(f"Analyzing SNP/Gene: {gene}...")
        
        # Treatment: Binarize Gene Expression (High vs Low)
        # 1 if > median, 0 else
        t_raw = df[gene].values
        t = (t_raw > np.median(t_raw)).astype(int)
        
        # Run models
        ates = run_causal_models(X, t, y)
        
        # Check ATE > 0.01 condition
        # We check if *any* model gives ATE > 0.01 (magnitude? or positive?)
        # Prompt: "SNP位点ATE大于0.01" -> ATE > 0.01
        
        row = {'SNP_Site': gene}
        keep = False
        for model_name, ate_val in ates.items():
            row[model_name] = ate_val
            if not np.isnan(ate_val) and abs(ate_val) > 0.01:
                keep = True
        
        if keep:
            final_results.append(row)
            
    # Output
    results_df = pd.DataFrame(final_results)
    
    # Use 'results' directory for consistency
    output_dir = r"e:\TERM\results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\nAnalysis Complete.")
    if not results_df.empty:
        model_cols = [c for c in results_df.columns if c != 'SNP_Site']
        
        # 1. Save separate files for each model (ATE > 0.01)
        for model in model_cols:
            model_mask = results_df[model].fillna(0).abs() > 0.01
            model_df = results_df[model_mask]
            
            model_out_path = os.path.join(output_dir, f"casual_{model}.csv")
            model_df.to_csv(model_out_path, index=False)
            print(f"Saved {model} results to {model_out_path} ({len(model_df)} genes)")
            
        # 2. Save Union (Already collected in results_df due to loop logic)
        union_out_path = os.path.join(output_dir, "casual_union.csv")
        results_df.to_csv(union_out_path, index=False)
        print(f"Saved Union results to {union_out_path}")
        
        print("Union Preview:")
        print(results_df.head())
    else:
        print("No SNP sites found with ATE > 0.01 across models.")

if __name__ == "__main__":
    main()
