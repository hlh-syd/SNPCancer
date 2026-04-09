import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import logging
import warnings
from torch.nn.utils import spectral_norm
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

# Configuration
warnings.filterwarnings("ignore")

# Configuration
CONFIG = {
    'Z_DIM': 100,
    'H': 84,
    'W': 100,
    'BATCH_SIZE': 64,
    'LR': 0.0002,
    'GP_LAMBDA': 10.0,
    'LR_SCHED_STEP': 30,
    'LR_SCHED_GAMMA': 0.5,
    'LABEL_SMOOTHING': 0.1,
    'EPOCHS': int(os.environ.get('GAN_EPOCHS', '100')),  # Adjusted for demo/speed, paper might use more
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'DATA_PATH': r"e:\TERM\data\merged_dataset_normalized.tsv",
    'CANDIDATE_PATH': r"e:\TERM\results\snp_univariate_reliable.csv",
    'OUTPUT_DIR': r"e:\TERM\results"
}

# --- 1. Data Loading & Preprocessing ---

def load_and_preprocess():
    logger.info(f"Loading data from {CONFIG['DATA_PATH']}...")
    if not os.path.exists(CONFIG['DATA_PATH']):
        logger.error("Data file not found.")
        return None, None, None, None

    df = pd.read_csv(CONFIG['DATA_PATH'], sep="\t", index_col=0)
    
    # Identify Columns
    clinical_patterns = ['age', 'gender', 'stage', 'status', 'event', 'dead']
    # Explicitly check for 'E' or 'event' for label
    label_col = 'E' if 'E' in df.columns else None
    if not label_col:
        for c in df.columns:
            if c.lower() in ['event', 'status', 'dead']:
                label_col = c
                break
    
    if not label_col:
        logger.error("Could not find Event/Label column (E, Status, etc).")
        return None, None, None, None

    # Clinical Covariates (for ATE)
    clinical_cols = [c for c in df.columns if any(p in c.lower() for p in ['age', 'gender', 'stage'])]
    
    # Gene Features
    gene_cols = [c for c in df.columns if c.startswith("ENSG")]
    
    logger.info(f"Samples: {len(df)}, Genes: {len(gene_cols)}, Clinical: {len(clinical_cols)}")
    
    # Prepare X (Genes) and Y (Label)
    X_genes = df[gene_cols].values
    Y_labels = df[label_col].values
    X_clinical = df[clinical_cols].values
    
    # Reshape / Pad Genes to 84x100 = 8400
    target_dim = CONFIG['H'] * CONFIG['W']
    current_dim = X_genes.shape[1]
    
    if current_dim < target_dim:
        pad_width = target_dim - current_dim
        # Pad with zeros
        X_padded = np.pad(X_genes, ((0, 0), (0, pad_width)), 'constant')
    else:
        # Truncate if too large (unlikely given previous context)
        X_padded = X_genes[:, :target_dim]
        
    # Reshape to (N, 1, 84, 100) for 2D processing in Discriminator, 
    # but Generator uses 1D. We'll keep it flattened for Dataset and reshape inside models.
    
    return X_padded, Y_labels, X_clinical, gene_cols, df

# --- 2. Model Architecture (RS-CGAN) ---

class SoftThresholding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.threshold = nn.Parameter(torch.zeros(1, channels, 1), requires_grad=True)
    
    def forward(self, x):
        # x: (B, C, L)
        # Soft thresholding: sign(x) * max(|x| - tau, 0)
        return torch.mul(torch.sign(x), F.relu(torch.abs(x) - torch.abs(self.threshold)))

class ResidualSoftThresholdingBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.st = SoftThresholding(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.st(out)
        return out + residual

# --- DragonNet Architecture for ATE Estimation ---

class EpsilonLayer(nn.Module):
    def __init__(self):
        super(EpsilonLayer, self).__init__()
        self.epsilon = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        return self.epsilon * torch.ones_like(x)[:, 0:1]

class DragonNet(nn.Module):
    def __init__(self, input_dim, neurons_per_layer=200):
        super(DragonNet, self).__init__()
        # Shared Representation
        self.shared1 = nn.Linear(input_dim, neurons_per_layer)
        self.shared2 = nn.Linear(neurons_per_layer, neurons_per_layer)
        self.shared3 = nn.Linear(neurons_per_layer, neurons_per_layer)
        
        # Propensity Head
        self.t_pred = nn.Linear(neurons_per_layer, 1)
        
        # Outcome Head 0 (T=0)
        self.y0_h1 = nn.Linear(neurons_per_layer, neurons_per_layer // 2)
        self.y0_h2 = nn.Linear(neurons_per_layer // 2, neurons_per_layer // 2)
        self.y0_out = nn.Linear(neurons_per_layer // 2, 1)
        
        # Outcome Head 1 (T=1)
        self.y1_h1 = nn.Linear(neurons_per_layer, neurons_per_layer // 2)
        self.y1_h2 = nn.Linear(neurons_per_layer // 2, neurons_per_layer // 2)
        self.y1_out = nn.Linear(neurons_per_layer // 2, 1)
        
        # Epsilon
        self.epsilon = EpsilonLayer()
        
    def forward(self, x):
        # Shared
        x = F.elu(self.shared1(x))
        x = F.elu(self.shared2(x))
        x = F.elu(self.shared3(x))
        
        # Propensity
        t_p = torch.sigmoid(self.t_pred(x))
        
        # Head 0
        y0 = F.elu(self.y0_h1(x))
        y0 = F.elu(self.y0_h2(y0))
        y0_pred = self.y0_out(y0) 
        
        # Head 1
        y1 = F.elu(self.y1_h1(x))
        y1 = F.elu(self.y1_h2(y1))
        y1_pred = self.y1_out(y1)
        
        # Epsilon
        eps = self.epsilon(t_p)
        
        return y0_pred, y1_pred, t_p, eps

def dragonnet_loss(y_true, t_true, y0_pred, y1_pred, t_pred, eps, ratio=1.0):
    # Regression Loss
    loss0 = torch.sum((1 - t_true) * (y_true - y0_pred.squeeze())**2)
    loss1 = torch.sum(t_true * (y_true - y1_pred.squeeze())**2)
    
    # Classification Loss
    t_pred_clamp = torch.clamp(t_pred.squeeze(), 1e-6, 1 - 1e-6)
    loss_t = -torch.sum(t_true * torch.log(t_pred_clamp) + (1 - t_true) * torch.log(1 - t_pred_clamp))
    
    vanilla_loss = loss0 + loss1 + loss_t
    
    # Targeted Regularization
    h = (t_true / t_pred_clamp) - ((1 - t_true) / (1 - t_pred_clamp))
    
    y_pred = t_true * y1_pred.squeeze() + (1 - t_true) * y0_pred.squeeze()
    y_pert = y_pred + eps.squeeze() * h
    targeted_reg = torch.sum((y_true - y_pert)**2)
    
    return vanilla_loss + ratio * targeted_reg

class Generator(nn.Module):
    def __init__(self, z_dim, label_dim=1, out_h=84, out_w=100):
        super().__init__()
        self.h, self.w = out_h, out_w
        self.total_dim = out_h * out_w
        
        # Initial projection
        self.fc = nn.Linear(z_dim + label_dim, 256 * (self.total_dim // 64)) 
        # Simplified: Map to a lower dim sequence then upsample
        
        # 1D Processing with RS Blocks
        self.blocks = nn.Sequential(
            ResidualSoftThresholdingBlock(256),
            ResidualSoftThresholdingBlock(256)
        )
        
        self.final_conv = nn.Conv1d(256, 1, kernel_size=3, padding=1)
        # We need to ensure output size matches total_dim
        # Here we cheat slightly by projecting to exact size first if needed
        # Or we can reshape FC output to (B, 256, L')
        
        # Improved Architecture conforming to description:
        # 1D Conv -> Residual Soft Thresholding -> Output
        self.fc_input = nn.Linear(z_dim + label_dim, self.total_dim) # Direct map to full seq
        self.rstb = nn.Sequential(
             nn.Conv1d(1, 16, 3, padding=1),
             ResidualSoftThresholdingBlock(16),
             ResidualSoftThresholdingBlock(16),
             nn.Conv1d(16, 1, 3, padding=1)
        )
        
    def forward(self, z, labels):
        # z: (B, 100), labels: (B, 1)
        inp = torch.cat([z, labels], dim=1)
        x = self.fc_input(inp) # (B, 8400)
        x = x.view(x.size(0), 1, -1) # (B, 1, 8400)
        x = self.rstb(x) # (B, 1, 8400)
        x = x.view(x.size(0), 1, self.h, self.w) # (B, 1, 84, 100)
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_res = nn.Sequential(
            spectral_norm(nn.Conv2d(in_c, out_c, 3, padding=1)),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(out_c, out_c, 3, padding=1)),
            nn.BatchNorm2d(out_c)
        )
        
        self.att_mask = nn.Sequential(
            spectral_norm(nn.Conv2d(in_c, 1, 1)),
            nn.Sigmoid()
        )
        
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        
    def forward(self, x):
        # Attention Mechanism: (1 + M(x)) * F(x)
        mask = self.att_mask(x)
        res = self.conv_res(x)
        
        # Apply attention to residual branch
        attended_res = res * (1 + mask)
        
        return attended_res + self.shortcut(x)

class Discriminator(nn.Module):
    def __init__(self, label_dim=1):
        super().__init__()
        # Input: (B, 1+1, 84, 100) -> Image + Label Channel
        self.conv1 = spectral_norm(nn.Conv2d(2, 32, 3, padding=1))
        self.rab1 = ResidualAttentionBlock(32, 64)
        self.rab2 = ResidualAttentionBlock(64, 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1)
        
    def forward(self, x, labels):
        # x: (B, 1, 84, 100)
        # labels: (B, 1) -> expand to (B, 1, 84, 100)
        labels_map = labels.unsqueeze(2).unsqueeze(3).expand_as(x)
        inp = torch.cat([x, labels_map], dim=1)
        
        out = self.conv1(inp)
        out = self.rab1(out)
        out = F.max_pool2d(out, 2)
        out = self.rab2(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)



def compute_gradient_penalty(netD, real_samples, fake_samples, labels):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = netD(interpolates, labels)
    grads = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grads = grads.view(grads.size(0), -1)
    grad_norm = grads.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()

# --- 3. Training & ATE Estimation ---

def train_rs_cgan(X_train, Y_train):
    device = torch.device(CONFIG['DEVICE'])
    
    # Dataset
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).view(-1, 1, CONFIG['H'], CONFIG['W']),
        torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    )
    dataloader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    
    # Models
    netG = Generator(CONFIG['Z_DIM']).to(device)
    netD = Discriminator().to(device)
    
    # Optimizers
    optG = torch.optim.Adam(netG.parameters(), lr=CONFIG['LR'], betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=CONFIG['LR'], betas=(0.5, 0.999))
    schedG = torch.optim.lr_scheduler.StepLR(optG, step_size=CONFIG['LR_SCHED_STEP'], gamma=CONFIG['LR_SCHED_GAMMA'])
    schedD = torch.optim.lr_scheduler.StepLR(optD, step_size=CONFIG['LR_SCHED_STEP'], gamma=CONFIG['LR_SCHED_GAMMA'])
    
    # Losses
    criterion_adv = nn.BCEWithLogitsLoss()
    
    logger.info("Starting RS-CGAN Training...")
    
    for epoch in range(CONFIG['EPOCHS']):
        for i, (real_imgs, labels) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            
            # --- Train Discriminator ---
            optD.zero_grad()
            
            # Real
            pred_real = netD(real_imgs, labels)
            loss_real = criterion_adv(pred_real, torch.ones_like(pred_real))
            
            # Fake
            z = torch.randn(batch_size, CONFIG['Z_DIM']).to(device)
            fake_imgs = netG(z, labels)
            pred_fake = netD(fake_imgs.detach(), labels)
            loss_fake = criterion_adv(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optD.step()
            
            # --- Train Generator ---
            optG.zero_grad()
            
            pred_fake_G = netD(fake_imgs, labels)
            loss_G_adv = criterion_adv(pred_fake_G, torch.ones_like(pred_fake_G))
            
            # Distance Similarity Penalty (Feature Matching / Moment Matching)
            # Simplified: L1 between mean of fake and real batch
            loss_G_dist = F.l1_loss(fake_imgs.mean(dim=0), real_imgs.mean(dim=0))
            
            loss_G = loss_G_adv + 10.0 * loss_G_dist # Lambda=10
            
            loss_G.backward()
            optG.step()
            
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{CONFIG['EPOCHS']}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}, GP: {gp.item():.4f}")

        schedG.step()
        schedD.step()
    
    return netG

def train_dragonnet(X, T, Y, epochs=100, batch_size=64, lr=1e-3, device='cpu'):
    # X: Features (Covariates)
    # T: Treatment (0/1)
    # Y: Outcome
    
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(T, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = DragonNet(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, t_batch, y_batch in dataloader:
            x_batch, t_batch, y_batch = x_batch.to(device), t_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y0_pred, y1_pred, t_pred, eps = model(x_batch)
            
            loss = dragonnet_loss(y_batch, t_batch, y0_pred, y1_pred, t_pred, eps)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
    return model

def estimate_ate_with_augmented_data(netG, X_real, Y_real, X_clinical, gene_cols, df_orig):
    logger.info("Generating synthetic data for ATE estimation...")
    device = torch.device(CONFIG['DEVICE'])
    netG.eval()
    
    # Generate Synthetic Data
    n_synth = len(X_real)
    z = torch.randn(n_synth, CONFIG['Z_DIM']).to(device)
    y_syn = torch.randint(0, 2, (n_synth, 1)).float().to(device)
    
    with torch.no_grad():
        x_syn = netG(z, y_syn).cpu().numpy().reshape(n_synth, -1)
    
    y_syn = y_syn.cpu().numpy().flatten()
    X_real_flat = X_real.reshape(len(X_real), -1)
    
    # Combine Real + Synthetic
    X_combined = np.vstack([X_real_flat, x_syn])
    Y_combined = np.concatenate([Y_real, y_syn])
    
    # Resample clinical data for synthetic samples
    clinical_indices = np.random.choice(len(X_clinical), n_synth, replace=True)
    X_clinical_syn = X_clinical[clinical_indices]
    X_clinical_combined = np.vstack([X_clinical, X_clinical_syn])
    
    # Load Candidates
    if os.path.exists(CONFIG['CANDIDATE_PATH']):
        cand_df = pd.read_csv(CONFIG['CANDIDATE_PATH'], sep="\t")
        if 'snp' in cand_df.columns:
            candidates = cand_df['snp'].tolist()
        else:
            candidates = cand_df.iloc[:, 0].tolist()
        candidates = [c for c in candidates if c in gene_cols]
        logger.info(f"Evaluating ATE for {len(candidates)} candidate genes using DragonNet...")
    else:
        logger.warning("Candidate file not found. Using top 20 genes.")
        candidates = gene_cols[:20]

    results = []
    gene_col_map = {name: i for i, name in enumerate(gene_cols)}
    
    for gene in candidates:
        if gene not in gene_col_map:
            continue
            
        idx = gene_col_map[gene]
        
        # Prepare Data for DragonNet
        # Treatment T: Gene Expression (Binarized High/Low)
        # Covariates X: Clinical Data ONLY (or + other Genes? Paper says adjust for confounders. Usually Clinical)
        # Outcome Y: Disease Status
        
        gene_vals = X_combined[:, idx]
        T = (gene_vals > np.median(gene_vals)).astype(int)
        W = X_clinical_combined # Confounders
        Y = Y_combined
        
        # Train DragonNet
        # Note: In a full pipeline, we might want to include PCA of other genes as confounders too.
        # For now, we use Clinical variables as W.
        
        try:
            model = train_dragonnet(W, T, Y, epochs=50, batch_size=CONFIG['BATCH_SIZE'], device=CONFIG['DEVICE'])
            
            # Predict ATE
            model.eval()
            with torch.no_grad():
                W_tensor = torch.tensor(W, dtype=torch.float32).to(CONFIG['DEVICE'])
                y0_pred, y1_pred, t_pred, _ = model(W_tensor)
                
                # ATE = E[Y1 - Y0]
                ate = (y1_pred - y0_pred).mean().item()
            
            results.append({
                'SNP_Site': gene,
                'ATE': ate,
                'Abs_ATE': abs(ate)
            })
            # logger.info(f"Gene {gene}: ATE={ate:.4f}")
            
        except Exception as e:
            logger.error(f"DragonNet training failed for {gene}: {e}")

    # Save Results
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values('Abs_ATE', ascending=False)
        out_path = os.path.join(CONFIG['OUTPUT_DIR'], 'rs_cgan_dragonnet_ate_results.csv')
        res_df.to_csv(out_path, index=False)
        logger.info(f"Saved DragonNet ATE results to {out_path}")
        print(res_df.head(15))
    else:
        logger.warning("No ATE results generated.")

# --- Main ---

def main():
    if not os.path.exists(CONFIG['OUTPUT_DIR']):
        os.makedirs(CONFIG['OUTPUT_DIR'])
        
    # 1. Load
    X, Y, X_clin, gene_cols, df = load_and_preprocess()
    if X is None:
        return

    # 2. Train RS-CGAN
    netG = train_rs_cgan(X, Y)
    
    # 3. Estimate ATE
    estimate_ate_with_augmented_data(netG, X, Y, X_clin, gene_cols, df)

    # 4. Evaluation
    if evaluate_all_metrics:
        logger.info("Performing GAN Evaluation with Real Data...")
        netG.eval()
        with torch.no_grad():
            z = torch.randn(X.shape[0], CONFIG['Z_DIM']).to(CONFIG['DEVICE'])
            # Randomly sample labels for generation to match distribution or use real labels
            # Using real labels to condition generation is fair for conditional GANs
            y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1).to(CONFIG['DEVICE'])
            
            # Generator expects (z, labels)
            fake_data = netG(z, y_tensor)
            
            # RS-CGAN Generator output is (B, 1, 84, 100) -> Flatten to match X (N, 8400)
            fake_data_np = fake_data.cpu().numpy().reshape(X.shape[0], -1)
            real_data_np = X.reshape(X.shape[0], -1)
            
            metrics = evaluate_all_metrics(real_data_np, fake_data_np, device=CONFIG['DEVICE'])
            logger.info(f"Final Evaluation Metrics: {metrics}")

if __name__ == "__main__":
    main()