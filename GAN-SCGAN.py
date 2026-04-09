import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import warnings
import importlib.util

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Evaluation Module
try:
    spec = importlib.util.spec_from_file_location("gan_eval", r"e:\TERM\Data_augmentation_evaluation.py")
    gan_eval = importlib.util.module_from_spec(spec)
    sys.modules["gan_eval"] = gan_eval
    spec.loader.exec_module(gan_eval)
    from gan_eval import evaluate_all_metrics
    logger.info("Successfully imported evaluate_all_metrics.")
except Exception as e:
    logger.error(f"Failed to import evaluation module: {e}")
    evaluate_all_metrics = None

# --- 1. Path Setup ---
PROJECT_ROOT = r"e:\TERM"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Import Causal Forest
HAS_FOREST = False
try:
    from causalml.inference.tree import CausalRandomForestRegressor
    HAS_FOREST = True
    logger.info("Successfully imported CausalRandomForestRegressor.")
except ImportError:
    try:
        from causalml.inference.tree.causal.causalforest import CausalRandomForestRegressor
        HAS_FOREST = True
        logger.info("Successfully imported CausalRandomForestRegressor from local source.")
    except ImportError:
        logger.error("Failed to import CausalRandomForestRegressor.")

# Configuration
CONFIG = {
    'Z_DIM': 100,
    'H': 84,
    'W': 100,
    'BATCH_SIZE': 64,
    'LR': 0.0002,
    'EPOCHS': 100,  # Match RS-CGAN
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'DATA_PATH': r"e:\TERM\data\merged_dataset_normalized.tsv",
    'CANDIDATE_PATH': r"e:\TERM\results\snp_univariate_reliable.csv",
    'OUTPUT_DIR': r"e:\TERM\results",
    'LAMBDA_SPARSE': 0.1,
    'LAMBDA_LAYER': 0.1
}

# --- 2. Data Loading ---
def load_and_preprocess():
    logger.info(f"Loading data from {CONFIG['DATA_PATH']}...")
    if not os.path.exists(CONFIG['DATA_PATH']):
        logger.error("Data file not found.")
        return None, None, None, None, None

    df = pd.read_csv(CONFIG['DATA_PATH'], sep="\t", index_col=0)
    
    # Clinical Covariates
    clinical_cols = [c for c in df.columns if any(p in c.lower() for p in ['age', 'gender', 'stage'])]
    gene_cols = [c for c in df.columns if c.startswith("ENSG")]
    
    label_col = 'E' if 'E' in df.columns else None
    if not label_col:
        for c in df.columns:
            if c.lower() in ['event', 'status', 'dead']:
                label_col = c
                break
    
    if not label_col:
        logger.error("Could not find Event/Label column.")
        return None, None, None, None, None

    logger.info(f"Samples: {len(df)}, Genes: {len(gene_cols)}, Clinical: {len(clinical_cols)}")
    
    X_genes = df[gene_cols].values
    Y_labels = df[label_col].values
    X_clinical = df[clinical_cols].values
    
    target_dim = CONFIG['H'] * CONFIG['W']
    current_dim = X_genes.shape[1]
    
    if current_dim < target_dim:
        pad_width = target_dim - current_dim
        X_padded = np.pad(X_genes, ((0, 0), (0, pad_width)), 'constant')
    else:
        X_padded = X_genes[:, :target_dim]
        
    return X_padded, Y_labels, X_clinical, gene_cols, df

# --- 3. Model Architecture (Enhanced SCGAN) ---

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn_map = self.sigmoid(self.conv(x))
        return x * attn_map, attn_map

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.att_mask = nn.Sequential(
            nn.Conv2d(in_c, 1, 1),
            nn.Sigmoid()
        )
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        
    def forward(self, x):
        mask = self.att_mask(x)
        res = self.conv_res(x)
        attended_res = res * (1 + mask)
        return attended_res + self.shortcut(x)

class Generator(nn.Module):
    def __init__(self, z_dim, label_dim=1):
        super(Generator, self).__init__()
        self.init_h, self.init_w = 21, 25
        self.fc = nn.Linear(z_dim + label_dim, 128 * self.init_h * self.init_w)
        
        # Deep Residual Generator with Upsampling
        self.main = nn.Sequential(
            ResidualBlock(128),
            nn.Upsample(scale_factor=2), # 42x50
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Upsample(scale_factor=2), # 84x100
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32)
        )
        
        self.attention = SpatialAttention(32)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
    def forward(self, z, labels):
        inp = torch.cat([z, labels], dim=1)
        x = self.fc(inp)
        x = x.view(-1, 128, self.init_h, self.init_w)
        x = self.main(x)
        x, attn_map = self.attention(x)
        out = self.final_conv(x)
        return out, attn_map, x

class Discriminator(nn.Module):
    def __init__(self, label_dim=1):
        super(Discriminator, self).__init__()
        # Enhanced Discriminator using Residual Attention Blocks (similar to RS-CGAN)
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.rab1 = ResidualAttentionBlock(32, 64)
        self.rab2 = ResidualAttentionBlock(64, 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1)
        
    def forward(self, x, labels, return_features=False):
        labels_map = labels.unsqueeze(2).unsqueeze(3).expand_as(x)
        inp = torch.cat([x, labels_map], dim=1)
        
        feat1 = self.conv1(inp)
        feat2 = self.rab1(feat1)
        feat2_pooled = F.max_pool2d(feat2, 2)
        feat3 = self.rab2(feat2_pooled)
        out = self.pool(feat3)
        out = out.view(out.size(0), -1)
        validity = self.fc(out)
        
        if return_features:
            return validity, [feat1, feat2, feat3]
        return validity

# --- 4. Training ---

def train_scgan(X_train, Y_train):
    device = torch.device(CONFIG['DEVICE'])
    X_min = X_train.min()
    X_max = X_train.max()
    X_scaled = (X_train - X_min) / (X_max - X_min + 1e-8)
    
    dataset = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32).view(-1, 1, CONFIG['H'], CONFIG['W']),
        torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    )
    dataloader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    
    netG = Generator(CONFIG['Z_DIM']).to(device)
    netD = Discriminator().to(device)
    
    optG = torch.optim.Adam(netG.parameters(), lr=CONFIG['LR'], betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=CONFIG['LR'], betas=(0.5, 0.999))
    
    criterion = nn.BCEWithLogitsLoss()
    
    logger.info(f"Starting Enhanced SCGAN Training for {CONFIG['EPOCHS']} epochs...")
    
    for epoch in range(CONFIG['EPOCHS']):
        for i, (real_imgs, labels) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            
            # --- Train Discriminator ---
            optD.zero_grad()
            pred_real = netD(real_imgs, labels)
            loss_real = criterion(pred_real, torch.ones_like(pred_real))
            
            z = torch.randn(batch_size, CONFIG['Z_DIM']).to(device)
            fake_imgs, _, _ = netG(z, labels)
            pred_fake = netD(fake_imgs.detach(), labels)
            loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optD.step()
            
            # --- Train Generator ---
            optG.zero_grad()
            
            pred_fake_G, fake_feats = netD(fake_imgs, labels, return_features=True)
            _, real_feats = netD(real_imgs, labels, return_features=True)
            
            loss_G_adv = criterion(pred_fake_G, torch.ones_like(pred_fake_G))
            loss_G_sparse = torch.mean(torch.abs(fake_imgs))
            
            loss_G_layer = 0
            # Align feature list lengths if needed (though here they should match)
            for f_fake, f_real in zip(fake_feats, real_feats):
                 loss_G_layer += F.mse_loss(f_fake, f_real.detach())
            
            loss_G = loss_G_adv + CONFIG['LAMBDA_SPARSE'] * loss_G_sparse + CONFIG['LAMBDA_LAYER'] * loss_G_layer
            
            loss_G.backward()
            optG.step()
            
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{CONFIG['EPOCHS']}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
            
    return netG, X_min, X_max

# --- 5. Causal Forest Inference ---

def estimate_cate_with_causal_forest(netG, X_real, Y_real, X_clinical, gene_cols, x_min, x_max):
    if not HAS_FOREST:
        logger.error("Causal Forest not available. Skipping CATE estimation.")
        return

    logger.info("Estimating CATE using Causal Forest (causalml)...")
    device = torch.device(CONFIG['DEVICE'])
    netG.eval()
    
    # 1. Augment Data
    n_synth = len(X_real)
    z = torch.randn(n_synth, CONFIG['Z_DIM']).to(device)
    y_syn = torch.randint(0, 2, (n_synth, 1)).float().to(device)
    
    with torch.no_grad():
        x_syn, _, _ = netG(z, y_syn)
        x_syn = x_syn.cpu().numpy().reshape(n_synth, -1)
    
    x_syn = x_syn * (x_max - x_min + 1e-8) + x_min
    y_syn = y_syn.cpu().numpy().flatten()
    
    X_real_flat = X_real.reshape(len(X_real), -1)
    X_comb = np.vstack([X_real_flat, x_syn])
    Y_comb = np.concatenate([Y_real, y_syn])
    
    clin_idx = np.random.choice(len(X_clinical), n_synth, replace=True)
    X_clin_syn = X_clinical[clin_idx]
    X_clin_comb = np.vstack([X_clinical, X_clin_syn])
    
    # 2. Load Candidates
    if os.path.exists(CONFIG['CANDIDATE_PATH']):
        cand_df = pd.read_csv(CONFIG['CANDIDATE_PATH'], sep="\t")
        if 'snp' in cand_df.columns:
            candidates = cand_df['snp'].tolist()
        else:
            candidates = cand_df.iloc[:, 0].tolist()
        candidates = [c for c in candidates if c in gene_cols]
    else:
        candidates = gene_cols[:20]
        
    gene_map = {g: i for i, g in enumerate(gene_cols)}
    results = []
    
    # 3. Causal Forest
    # Parameters for CausalRandomForestRegressor
    # Note: control_name is not strictly needed if we pass treatment vector manually
    
    for gene in candidates:
        if gene not in gene_map: continue
        idx = gene_map[gene]
        
        # Treatment: High vs Low
        T = (X_comb[:, idx] > np.median(X_comb[:, idx])).astype(int)
        W = X_clin_comb # Covariates
        Y = Y_comb # Outcome
        
        try:
            # Initialize Causal Forest
            # n_estimators=100 matches the 'complexity' request roughly (standard RF size)
            cf = CausalRandomForestRegressor(n_estimators=100, criterion='mse', n_jobs=1)
            
            # Fit
            # X=Covariates, treatment=Treatment Vector, y=Outcome
            cf.fit(X=W, treatment=T, y=Y)
            
            # Estimate CATE
            cate_estimates = cf.predict(X=W)
            
            mean_cate = np.mean(cate_estimates)
            mean_abs_cate = np.mean(np.abs(cate_estimates))
            
            # Calculate simple confidence interval (std based)
            # causalml's predict doesn't always return CI by default unless specified or supported by version
            # We'll use std of estimates as a proxy for heterogeneity
            std_cate = np.std(cate_estimates)
            
            results.append({
                'SNP_Site': gene,
                'Mean_CATE': mean_cate,
                'Mean_Abs_CATE': mean_abs_cate,
                'Std_CATE': std_cate
            })
            
            logger.info(f"Gene {gene}: Mean CATE={mean_cate:.4f}")
            
        except Exception as e:
            logger.error(f"Causal Forest failed for {gene}: {e}")
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values('Mean_Abs_CATE', ascending=False)
        out_path = os.path.join(CONFIG['OUTPUT_DIR'], 'scgan_causal_forest_results.csv')
        res_df.to_csv(out_path, index=False)
        logger.info(f"Saved Causal Forest results to {out_path}")
        print(res_df.head(15))

def main():
    if not os.path.exists(CONFIG['OUTPUT_DIR']):
        os.makedirs(CONFIG['OUTPUT_DIR'])
        
    X, Y, X_clin, gene_cols, _ = load_and_preprocess()
    if X is None: return
    
    netG, xmin, xmax = train_scgan(X, Y)
    estimate_cate_with_causal_forest(netG, X, Y, X_clin, gene_cols, xmin, xmax)

    # 6. Evaluation
    if evaluate_all_metrics:
        logger.info("Performing GAN Evaluation with Real Data...")
        netG.eval()
        with torch.no_grad():
            z = torch.randn(X.shape[0], CONFIG['Z_DIM']).to(CONFIG['DEVICE'])
            y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1).to(CONFIG['DEVICE'])
            
            fake_imgs, _, _ = netG(z, y_tensor)
            fake_imgs_np = fake_imgs.cpu().numpy().reshape(X.shape[0], -1)
            
            # Inverse transform to original space
            fake_data_orig = fake_imgs_np * (xmax - xmin + 1e-8) + xmin
            
            real_data_np = X.reshape(X.shape[0], -1)
            
            metrics = evaluate_all_metrics(real_data_np, fake_data_orig, device=CONFIG['DEVICE'])
            logger.info(f"Final Evaluation Metrics: {metrics}")

if __name__ == "__main__":
    main()
