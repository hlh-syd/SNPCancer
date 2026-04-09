import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from statsmodels.stats.multitest import fdrcorrection
import logging
import warnings
import matplotlib.pyplot as plt
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor

# --- 1. Path & Env Setup ---
PROJECT_ROOT = r"e:\TERM"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

CONFIG = {
    'Z_DIM': 100,
    'H': 84,
    'W': 100,
    'BATCH_SIZE': 64,
    'LR': 0.0002,
    'EPOCHS': 50,  # 优化后收敛更快，可减少Epochs
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'DATA_PATH': r"e:\TERM\data\merged_dataset_normalized.tsv",
    'OUTPUT_DIR': r"e:\TERM\results\optimized_pipeline",
    'LAMBDA_SPARSE': 0.1,
    'LAMBDA_LAYER': 0.1,
    'NUM_WORKERS': 4 # 并发计算SNP数量
}

os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# --- 2. Statistical Preprocessing & Dimensionality Reduction ---

def fast_hwe_filter(X_genes, gene_cols, threshold=1e-6):
    """ Round 1: Vectorized Hardy-Weinberg Equilibrium test. """
    logger.info("Running vectorized HWE test...")
    X_tensor = torch.tensor(X_genes, dtype=torch.float32)
    N = X_tensor.shape[0]
    
    n_het = torch.sum(X_tensor == 1, dim=0)
    n_hom_alt = torch.sum(X_tensor == 2, dim=0)
    
    p = (n_het + 2 * n_hom_alt) / (2 * N)
    exp_het = 2 * p * (1 - p) * N
    
    chi2 = (n_het - exp_het)**2 / (exp_het + 1e-8)
    # Approx P-value (1 df)
    # Using scipy for exact chi2 survival function is safer
    chi2_np = chi2.cpu().numpy()
    p_vals = stats.chi2.sf(chi2_np, df=1)
    
    keep_mask = p_vals >= threshold
    filtered_genes = [g for i, g in enumerate(gene_cols) if keep_mask[i]]
    
    logger.info(f"HWE Filter: kept {len(filtered_genes)}/{len(gene_cols)} SNPs (P >= {threshold}).")
    return X_genes[:, keep_mask], filtered_genes

def fast_ld_pruning(X_genes, gene_cols, r2_threshold=0.8):
    """ Round 2: GPU-accelerated Linkage Disequilibrium Pruning. """
    logger.info("Running GPU-accelerated LD Pruning...")
    # For very large matrices, we use simple correlation on GPU
    X_t = torch.tensor(X_genes, dtype=torch.float16, device=CONFIG['DEVICE'])
    X_norm = (X_t - X_t.mean(dim=0)) / (X_t.std(dim=0) + 1e-8)
    
    # We compute chunks to avoid OOM
    M = X_norm.shape[1]
    keep_indices = []
    dropped = set()
    
    # Simplified greedy pruning (in practice use sliding window)
    for i in range(min(2000, M)): # Just scanning top 2000 for demonstration speed
        if i in dropped: continue
        keep_indices.append(i)
        # Compute correlation of i with all remaining
        corr = torch.abs(torch.matmul(X_norm[:, i:i+1].T, X_norm[:, i:])) / X_norm.shape[0]
        high_ld = (corr > r2_threshold).squeeze().nonzero().squeeze().tolist()
        if isinstance(high_ld, int): high_ld = [high_ld]
        for idx in high_ld:
            real_idx = i + idx
            if real_idx != i:
                dropped.add(real_idx)
                
    # Add the rest that weren't scanned
    for i in range(2000, M):
        if i not in dropped: keep_indices.append(i)
        
    keep_indices = sorted(list(set(keep_indices)))
    filtered_genes = [gene_cols[i] for i in keep_indices]
    
    logger.info(f"LD Pruning: kept {len(filtered_genes)}/{len(gene_cols)} Tag SNPs.")
    return X_genes[:, keep_indices], filtered_genes

def population_stratification_pca(X_genes, X_clinical):
    """ Round 3: Randomized SVD for PCA to adjust population stratification. """
    logger.info("Computing Top 10 PCs for Population Stratification adjustment...")
    svd = TruncatedSVD(n_components=10, algorithm='randomized', random_state=42)
    PCs = svd.fit_transform(X_genes)
    # Append PCs to clinical covariates
    X_clinical_adj = np.hstack([X_clinical, PCs])
    logger.info("Added 10 PCs to clinical covariates.")
    return X_clinical_adj, PCs

def fdr_screening(X_genes, Y_labels, gene_cols):
    """ Round 4: Vectorized scoring and FDR correction. """
    logger.info("Running fast univariate screening with FDR correction...")
    # Fast vectorized Pearson correlation as a proxy for Score Test
    X_norm = (X_genes - X_genes.mean(axis=0)) / (X_genes.std(axis=0) + 1e-8)
    Y_norm = (Y_labels - Y_labels.mean()) / (Y_labels.std() + 1e-8)
    
    r = np.dot(Y_norm, X_norm) / len(Y_labels)
    # T-statistic
    t_stat = r * np.sqrt((len(Y_labels) - 2) / (1 - r**2 + 1e-8))
    p_vals = 2 * stats.t.sf(np.abs(t_stat), df=len(Y_labels)-2)
    
    # Plot QQ before FDR
    plot_qq(p_vals, "QQ-Plot (Before FDR)", os.path.join(CONFIG['OUTPUT_DIR'], "qq_plot_initial.png"))
    
    reject, pvals_corrected = fdrcorrection(p_vals, alpha=0.05, method='indep')
    significant_genes = [g for i, g in enumerate(gene_cols) if reject[i]]
    
    # Fallback if too strict
    if len(significant_genes) < 5:
        logger.warning("FDR too strict, falling back to top 50 raw p-values.")
        top_idx = np.argsort(p_vals)[:50]
        significant_genes = [gene_cols[i] for i in top_idx]
        
    logger.info(f"Screening selected {len(significant_genes)} candidates.")
    return significant_genes

def plot_qq(p_vals, title, out_path):
    p_vals = np.clip(p_vals, 1e-15, 1.0)
    obs = -np.log10(np.sort(p_vals))
    exp = -np.log10(np.arange(1, len(p_vals) + 1) / (len(p_vals) + 1))
    
    # Calculate Lambda
    chi2_obs = stats.chi2.ppf(1 - p_vals, 1)
    lambda_gc = np.median(chi2_obs) / 0.4549
    
    plt.figure(figsize=(6,6))
    plt.scatter(exp, obs, s=10, c='b', alpha=0.5)
    plt.plot([0, max(exp)], [0, max(exp)], 'r--')
    plt.title(f"{title}\n$\lambda_{{GC}}$ = {lambda_gc:.3f}")
    plt.xlabel("Expected $-\log_{10}(P)$")
    plt.ylabel("Observed $-\log_{10}(P)$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# --- 3. Optimized Deep Learning Models ---

class Generator(nn.Module):
    def __init__(self, z_dim, label_dim=1):
        super().__init__()
        self.init_h, self.init_w = 21, 25
        self.fc = nn.Linear(z_dim + label_dim, 32 * self.init_h * self.init_w)
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid() # Normalize to 0-1
        )
    def forward(self, z, labels):
        x = self.fc(torch.cat([z, labels], dim=1)).view(-1, 32, self.init_h, self.init_w)
        return self.main(x), None, None

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, 1)
    def forward(self, x, labels, return_features=False):
        labels_map = labels.unsqueeze(2).unsqueeze(3).expand(-1, 1, x.size(2), x.size(3))
        feat = self.main(torch.cat([x, labels_map], dim=1)).view(x.size(0), -1)
        out = self.fc(feat)
        if return_features: return out, [feat]
        return out

class CausalTransformerSCM(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=2, num_layers=1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head_t = nn.Linear(d_model, 1)
        self.head_y0 = nn.Linear(d_model, 1)
        self.head_y1 = nn.Linear(d_model, 1)
        self.eps = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        x_emb = self.proj(x).unsqueeze(1)
        rep = self.transformer(x_emb).squeeze(1)
        t_prob = torch.sigmoid(self.head_t(rep))
        return self.head_y0(rep), self.head_y1(rep), t_prob, self.eps * torch.ones_like(t_prob)

def dragonnet_loss(y_true, t_true, y0_pred, y1_pred, t_pred, eps):
    t_true = t_true.view(-1, 1)
    y_true = y_true.view(-1, 1)
    loss0 = torch.sum((1 - t_true) * (y_true - y0_pred)**2)
    loss1 = torch.sum(t_true * (y_true - y1_pred)**2)
    t_pred_c = torch.clamp(t_pred, 1e-4, 1 - 1e-4)
    loss_t = -torch.sum(t_true * torch.log(t_pred_c) + (1 - t_true) * torch.log(1 - t_pred_c))
    y_pred = t_true * y1_pred + (1 - t_true) * y0_pred
    h = (t_true / t_pred_c) - ((1 - t_true) / (1 - t_pred_c))
    reg = torch.sum((y_true - (y_pred + eps * h))**2)
    return loss0 + loss1 + loss_t + 0.1 * reg

# --- 4. Round 5: Mixed Precision Training (AMP) ---

def train_scgan_amp(X_train, Y_train):
    logger.info("Training SCGAN with Automatic Mixed Precision (AMP)...")
    device = torch.device(CONFIG['DEVICE'])
    scaler = torch.cuda.amp.GradScaler() # AMP Scaler
    
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).view(-1, 1, CONFIG['H'], CONFIG['W']),
        torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    )
    dataloader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    
    netG = Generator(CONFIG['Z_DIM']).to(device)
    netD = Discriminator().to(device)
    optG = torch.optim.Adam(netG.parameters(), lr=CONFIG['LR'])
    optD = torch.optim.Adam(netD.parameters(), lr=CONFIG['LR'])
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(CONFIG['EPOCHS'] // 2): # Train less epochs for speed in demo
        for real_imgs, labels in dataloader:
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            batch_size = real_imgs.size(0)
            
            # Train D
            optD.zero_grad()
            with torch.cuda.amp.autocast():
                pred_real = netD(real_imgs, labels)
                z = torch.randn(batch_size, CONFIG['Z_DIM']).to(device)
                fake_imgs, _, _ = netG(z, labels)
                pred_fake = netD(fake_imgs.detach(), labels)
                loss_D = (criterion(pred_real, torch.ones_like(pred_real)) + 
                          criterion(pred_fake, torch.zeros_like(pred_fake))) / 2
            scaler.scale(loss_D).backward()
            scaler.step(optD)
            
            # Train G
            optG.zero_grad()
            with torch.cuda.amp.autocast():
                pred_fake_G, _ = netD(fake_imgs, labels, return_features=True)
                loss_G = criterion(pred_fake_G, torch.ones_like(pred_fake_G))
            scaler.scale(loss_G).backward()
            scaler.step(optG)
            scaler.update()
            
    return netG

def train_scm_amp(W, T, Y):
    device = torch.device(CONFIG['DEVICE'])
    model = CausalTransformerSCM(W.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # Added L2 Reg
    scaler = torch.cuda.amp.GradScaler()
    
    dataset = TensorDataset(torch.tensor(W, dtype=torch.float32), 
                            torch.tensor(T, dtype=torch.float32), 
                            torch.tensor(Y, dtype=torch.float32))
    dl = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    for _ in range(20):
        for w_b, t_b, y_b in dl:
            w_b, t_b, y_b = w_b.to(device), t_b.to(device), y_b.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                y0, y1, t_p, eps = model(w_b)
                loss = dragonnet_loss(y_b, t_b, y0, y1, t_p, eps)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        w_t = torch.tensor(W, dtype=torch.float32).to(device)
        y0, y1, _, _ = model(w_t)
        cate = (y1 - y0).cpu().numpy().flatten()
    return np.mean(cate), np.std(cate)

# --- 5. Round 6: Distributed / Asynchronous Inference ---

def estimate_cate_parallel(netG, X_real, Y_real, X_clinical, gene_cols, candidates):
    logger.info(f"Round 6: Asynchronous CATE estimation for {len(candidates)} SNPs...")
    # Generate Synthetic data once
    device = torch.device(CONFIG['DEVICE'])
    netG.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        z = torch.randn(len(X_real), CONFIG['Z_DIM']).to(device)
        y_syn = torch.randint(0, 2, (len(X_real), 1)).float().to(device)
        x_syn, _, _ = netG(z, y_syn)
        x_syn = x_syn.cpu().numpy().reshape(len(X_real), -1)
    
    X_comb = np.vstack([X_real.reshape(len(X_real), -1), x_syn])
    Y_comb = np.concatenate([Y_real, y_syn.cpu().numpy().flatten()])
    X_clin_comb = np.vstack([X_clinical, X_clinical]) # Simplified augmentation
    
    scaler = StandardScaler()
    W_norm = scaler.fit_transform(X_clin_comb)
    gene_map = {g: i for i, g in enumerate(gene_cols)}
    
    def worker(gene):
        idx = gene_map.get(gene)
        if idx is None: return None
        T = (X_comb[:, idx] > np.median(X_comb[:, idx])).astype(int)
        mean_cate, std_cate = train_scm_amp(W_norm, T, Y_comb)
        return {'SNP': gene, 'CATE': mean_cate, 'Std': std_cate, 'Abs_CATE': abs(mean_cate)}

    results = []
    with ThreadPoolExecutor(max_workers=CONFIG['NUM_WORKERS']) as executor:
        for res in executor.map(worker, candidates):
            if res: results.append(res)
            
    df_res = pd.DataFrame(results).sort_values('Abs_CATE', ascending=False)
    df_res.to_csv(os.path.join(CONFIG['OUTPUT_DIR'], 'final_cate_results.csv'), index=False)
    return df_res

# --- 6. Round 7: End-to-End Pipeline ---

class OptimizedCausalPipeline:
    def __init__(self):
        logger.info("Initializing Optimized Causal Pipeline...")
        
    def run(self):
        try:
            # 1. Load Data
            df = pd.read_csv(CONFIG['DATA_PATH'], sep="\t", index_col=0)
            clin_cols = [c for c in df.columns if 'stage' in c.lower() or 'age' in c.lower()]
            gene_cols = [c for c in df.columns if c.startswith("ENSG")]
            label_col = 'E' if 'E' in df.columns else df.columns[-1]
            
            X_genes = df[gene_cols].values
            Y_labels = df[label_col].values
            X_clin = df[clin_cols].values
            
            # Pad for SCGAN (84x100)
            target_dim = CONFIG['H'] * CONFIG['W']
            if X_genes.shape[1] < target_dim:
                X_padded = np.pad(X_genes, ((0,0), (0, target_dim - X_genes.shape[1])))
            else:
                X_padded = X_genes[:, :target_dim]
                gene_cols = gene_cols[:target_dim]

            # 2. HWE & LD
            X_genes, gene_cols = fast_hwe_filter(X_genes, gene_cols)
            X_genes, gene_cols = fast_ld_pruning(X_genes, gene_cols)
            
            # 3. PCA Stratification
            X_clin, _ = population_stratification_pca(X_genes, X_clin)
            
            # 4. FDR Screening
            candidates = fdr_screening(X_genes, Y_labels, gene_cols)
            
            # 5. AMP SCGAN
            netG = train_scgan_amp(X_padded, Y_labels)
            
            # 6. Parallel Transformer-SCM
            res_df = estimate_cate_parallel(netG, X_padded, Y_labels, X_clin, gene_cols, candidates)
            
            logger.info("Pipeline completed successfully!")
            print("\n=== Top 10 Causal SNPs ===")
            print(res_df.head(10).to_string(index=False))
            
        finally:
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    pipeline = OptimizedCausalPipeline()
    pipeline.run()
