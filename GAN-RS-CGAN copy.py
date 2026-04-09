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
from scipy.spatial.distance import cdist

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Configuration
CONFIG = {
    'Z_DIM': 100,
    'H': 84,
    'W': 100,
    'BATCH_SIZE': 64,
    'LR': 0.0002,
    'EPOCHS': 100,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'DATA_PATH': r"e:\TERM\data\merged_dataset_normalized.tsv",
    'CANDIDATE_PATH': r"e:\TERM\results\snp_univariate_reliable.csv",
    'OUTPUT_DIR': r"e:\TERM\results"
}

# --- 1. GTN / G-Transformer Implementation (Dense Version) ---
# Re-implemented to avoid dependency on torch_geometric/torch_sparse

class GCNConvDense(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConvDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: (N, in_channels)
        # adj: (N, N) - Normalized adjacency matrix
        
        # Support = XW
        support = torch.mm(x, self.weight)
        # Output = A * Support
        output = torch.mm(adj, support)
        
        return output + self.bias

class GTConvDense(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(GTConvDense, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Weight for mixing adjacency matrices: (out_channels, in_channels)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, A):
        # A: (in_channels, N, N) tensor of adjacency matrices
        # Weights: Softmax over input channels
        filter_weights = F.softmax(self.weight, dim=1) # (out, in)
        
        # Linear combination of As
        # Output: (out_channels, N, N)
        # Einsum: o=out, i=in, n=node, m=node
        # res[o, n, m] = sum_i(weight[o, i] * A[i, n, m])
        res = torch.einsum('oi,inm->onm', filter_weights, A)
        return res

class GTLayerDense(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, first=True):
        super(GTLayerDense, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        
        if self.first:
            self.conv1 = GTConvDense(in_channels, out_channels, num_nodes)
            self.conv2 = GTConvDense(in_channels, out_channels, num_nodes)
        else:
            self.conv1 = GTConvDense(in_channels, out_channels, num_nodes)

    def forward(self, A, H_=None):
        if self.first:
            result_A = self.conv1(A) # (out, N, N)
            result_B = self.conv2(A) # (out, N, N)
        else:
            result_A = H_
            result_B = self.conv1(A)
            
        # Matrix Multiplication of graphs (Meta-path learning)
        # H[i] = A[i] @ B[i]
        H = torch.bmm(result_A, result_B) # (out, N, N)
        return H

class GTN(nn.Module):
    def __init__(self, num_edge_types, num_channels, w_in, w_out, num_nodes, num_layers=2):
        super(GTN, self).__init__()
        self.num_edge_types = num_edge_types
        self.num_channels = num_channels # Number of GTN channels (new graph types learned)
        self.num_layers = num_layers
        self.w_in = w_in
        self.w_out = w_out
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GTLayerDense(num_edge_types, num_channels, num_nodes, first=True))
            else:
                self.layers.append(GTLayerDense(num_channels, num_channels, num_nodes, first=False))
        
        # GCN for feature extraction on learned graphs
        self.gcn = GCNConvDense(w_in, w_out)
        
        # Norm layer
        self.norm_layer = nn.LayerNorm(w_out * num_channels)

    def normalize_adj(self, adj):
        # D^-0.5 * A * D^-0.5
        # Add self loop
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_mat = torch.diag(deg_inv_sqrt)
        
        return torch.mm(torch.mm(deg_mat, adj), deg_mat)

    def forward(self, A, X):
        # A: (num_edge_types, N, N)
        # X: (N, w_in)
        
        Hs = []
        H = None
        for i in range(self.num_layers):
            if i == 0:
                H = self.layers[i](A)
            else:
                H = self.layers[i](A, H)
        
        # H is now (num_channels, N, N) - the learned graphs
        
        # Apply GCN on each channel
        X_out = []
        for i in range(self.num_channels):
            adj = H[i]
            norm_adj = self.normalize_adj(adj)
            x_gcn = F.relu(self.gcn(X, norm_adj))
            X_out.append(x_gcn)
            
        # Concat all channel outputs
        X_final = torch.cat(X_out, dim=1) # (N, num_channels * w_out)
        return X_final

# --- 2. Data Loading & Preprocessing ---

def load_and_preprocess():
    logger.info(f"Loading data from {CONFIG['DATA_PATH']}...")
    if not os.path.exists(CONFIG['DATA_PATH']):
        logger.error("Data file not found.")
        return None, None, None, None
    
    df = pd.read_csv(CONFIG['DATA_PATH'], sep="\t", index_col=0)
    
    clinical_patterns = ['age', 'gender', 'stage', 'status', 'event', 'dead']
    label_col = 'E' if 'E' in df.columns else None
    if not label_col:
        for c in df.columns:
            if c.lower() in ['event', 'status', 'dead']:
                label_col = c
                break
    
    if not label_col:
        logger.error("Could not find Event/Label column.")
        return None, None, None, None

    clinical_cols = [c for c in df.columns if any(p in c.lower() for p in ['age', 'gender', 'stage'])]
    gene_cols = [c for c in df.columns if c.startswith("ENSG")]
    
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

# --- 3. Graph Construction (SCM-like) ---

def build_adjacency_matrices(X_clinical, X_genes, threshold=0.5):
    # 1. Clinical Similarity Graph (e.g., Euclidean distance on normalized clinical data)
    scaler = StandardScaler()
    X_clin_norm = scaler.fit_transform(X_clinical)
    dist_clin = cdist(X_clin_norm, X_clin_norm, metric='euclidean')
    # Convert distance to similarity (Gaussian kernel)
    sim_clin = np.exp(-dist_clin ** 2 / 2.0)
    adj_clin = (sim_clin > threshold).astype(np.float32)
    
    # 2. Gene Correlation Graph (Simplify: use PCA or subset for speed if needed, here full corr)
    # Using a subset of genes or just random projection for speed as computing N*N correlation is fast for N=400
    # But here we want Patient-Patient similarity based on Genes
    dist_gene = cdist(X_genes, X_genes, metric='correlation')
    sim_gene = 1 - dist_gene
    adj_gene = (sim_gene > threshold).astype(np.float32)
    
    # 3. Identity (Self-loop mostly, base structure)
    adj_eye = np.eye(len(X_clinical), dtype=np.float32)
    
    # Stack: (3, N, N)
    adjs = np.stack([adj_clin, adj_gene, adj_eye], axis=0)
    return torch.tensor(adjs, dtype=torch.float32)

# --- 4. Model Architecture (RS-CGAN) ---

class SoftThresholding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.threshold = nn.Parameter(torch.zeros(1, channels, 1), requires_grad=True)
    def forward(self, x):
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

class Generator(nn.Module):
    def __init__(self, z_dim, label_dim=1, out_h=84, out_w=100):
        super().__init__()
        self.h, self.w = out_h, out_w
        self.total_dim = out_h * out_w
        self.fc_input = nn.Linear(z_dim + label_dim, self.total_dim)
        self.rstb = nn.Sequential(
             nn.Conv1d(1, 16, 3, padding=1),
             ResidualSoftThresholdingBlock(16),
             ResidualSoftThresholdingBlock(16),
             nn.Conv1d(16, 1, 3, padding=1)
        )
    def forward(self, z, labels):
        inp = torch.cat([z, labels], dim=1)
        x = self.fc_input(inp)
        x = x.view(x.size(0), 1, -1)
        x = self.rstb(x)
        x = x.view(x.size(0), 1, self.h, self.w)
        return x

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

class Discriminator(nn.Module):
    def __init__(self, label_dim=1):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.rab1 = ResidualAttentionBlock(32, 64)
        self.rab2 = ResidualAttentionBlock(64, 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1)
    def forward(self, x, labels):
        labels_map = labels.unsqueeze(2).unsqueeze(3).expand_as(x)
        inp = torch.cat([x, labels_map], dim=1)
        out = self.conv1(inp)
        out = self.rab1(out)
        out = F.max_pool2d(out, 2)
        out = self.rab2(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)

# --- 5. GTN-Enhanced DragonNet (SCM Causal Layer) ---

class EpsilonLayer(nn.Module):
    def __init__(self):
        super(EpsilonLayer, self).__init__()
        self.epsilon = nn.Parameter(torch.Tensor([0.0]))
    def forward(self, x):
        return self.epsilon * torch.ones_like(x)[:, 0:1]

class GTNDragonNet(nn.Module):
    def __init__(self, num_nodes, input_dim, neurons_per_layer=200):
        super(GTNDragonNet, self).__init__()
        
        # GTN Feature Extractor
        # num_edge_types=3 (Clin, Gene, Eye), num_channels=2 (Learn 2 meta-graphs)
        self.gtn = GTN(num_edge_types=3, num_channels=2, w_in=input_dim, w_out=64, num_nodes=num_nodes, num_layers=2)
        
        # DragonNet Heads
        # Input dim is GTN output dim: num_channels * w_out = 2 * 64 = 128
        self.repr_dim = 2 * 64
        
        self.shared1 = nn.Linear(self.repr_dim, neurons_per_layer)
        self.shared2 = nn.Linear(neurons_per_layer, neurons_per_layer)
        
        self.t_pred = nn.Linear(neurons_per_layer, 1)
        
        self.y0_h1 = nn.Linear(neurons_per_layer, neurons_per_layer // 2)
        self.y0_out = nn.Linear(neurons_per_layer // 2, 1)
        
        self.y1_h1 = nn.Linear(neurons_per_layer, neurons_per_layer // 2)
        self.y1_out = nn.Linear(neurons_per_layer // 2, 1)
        
        self.epsilon = EpsilonLayer()
        
    def forward(self, A, X):
        # GTN Pass -> Learning Structural Representation
        # This models the SCM dependencies
        gtn_embed = self.gtn(A, X) # (N, 128)
        
        # Standard DragonNet Flow
        x = F.elu(self.shared1(gtn_embed))
        x = F.elu(self.shared2(x))
        
        t_p = torch.sigmoid(self.t_pred(x))
        
        y0 = F.elu(self.y0_h1(x))
        y0_pred = self.y0_out(y0)
        
        y1 = F.elu(self.y1_h1(x))
        y1_pred = self.y1_out(y1)
        
        eps = self.epsilon(t_p)
        
        return y0_pred, y1_pred, t_p, eps

def dragonnet_loss(y_true, t_true, y0_pred, y1_pred, t_pred, eps, ratio=1.0):
    loss0 = torch.sum((1 - t_true) * (y_true - y0_pred.squeeze())**2)
    loss1 = torch.sum(t_true * (y_true - y1_pred.squeeze())**2)
    
    t_pred_clamp = torch.clamp(t_pred.squeeze(), 1e-6, 1 - 1e-6)
    loss_t = -torch.sum(t_true * torch.log(t_pred_clamp) + (1 - t_true) * torch.log(1 - t_pred_clamp))
    
    vanilla_loss = loss0 + loss1 + loss_t
    
    h = (t_true / t_pred_clamp) - ((1 - t_true) / (1 - t_pred_clamp))
    y_pred = t_true * y1_pred.squeeze() + (1 - t_true) * y0_pred.squeeze()
    y_pert = y_pred + eps.squeeze() * h
    targeted_reg = torch.sum((y_true - y_pert)**2)
    
    return vanilla_loss + ratio * targeted_reg

# --- 6. Training Functions ---

def train_rs_cgan(X_train, Y_train):
    device = torch.device(CONFIG['DEVICE'])
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).view(-1, 1, CONFIG['H'], CONFIG['W']),
        torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    )
    dataloader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    
    netG = Generator(CONFIG['Z_DIM']).to(device)
    netD = Discriminator().to(device)
    
    optG = torch.optim.Adam(netG.parameters(), lr=CONFIG['LR'], betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=CONFIG['LR'], betas=(0.5, 0.999))
    criterion_adv = nn.BCEWithLogitsLoss()
    
    logger.info("Starting RS-CGAN Training...")
    
    for epoch in range(CONFIG['EPOCHS']):
        for i, (real_imgs, labels) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            
            # Discriminator
            optD.zero_grad()
            pred_real = netD(real_imgs, labels)
            loss_real = criterion_adv(pred_real, torch.ones_like(pred_real))
            
            z = torch.randn(batch_size, CONFIG['Z_DIM']).to(device)
            fake_imgs = netG(z, labels)
            pred_fake = netD(fake_imgs.detach(), labels)
            loss_fake = criterion_adv(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optD.step()
            
            # Generator
            optG.zero_grad()
            pred_fake_G = netD(fake_imgs, labels)
            loss_G_adv = criterion_adv(pred_fake_G, torch.ones_like(pred_fake_G))
            loss_G_dist = F.l1_loss(fake_imgs.mean(dim=0), real_imgs.mean(dim=0))
            loss_G = loss_G_adv + 10.0 * loss_G_dist
            loss_G.backward()
            optG.step()
            
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{CONFIG['EPOCHS']}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
            
    return netG

def train_gtn_dragonnet(W, T, Y, adj_matrix, epochs=100, lr=1e-3, device='cpu'):
    # Note: GTN is a full-batch method usually, or requires complex sampling.
    # Given N is small (~800 after aug), we can use full batch.
    
    num_nodes = W.shape[0]
    input_dim = W.shape[1]
    
    model = GTNDragonNet(num_nodes=num_nodes, input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Move all data to device once
    W_t = torch.tensor(W, dtype=torch.float32).to(device)
    T_t = torch.tensor(T, dtype=torch.float32).to(device)
    Y_t = torch.tensor(Y, dtype=torch.float32).to(device)
    A_t = adj_matrix.to(device)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass with Graph Structure
        y0_pred, y1_pred, t_pred, eps = model(A_t, W_t)
        
        loss = dragonnet_loss(Y_t, T_t, y0_pred, y1_pred, t_pred, eps)
        loss.backward()
        optimizer.step()
        
    return model

def estimate_ate_with_gtn(netG, X_real, Y_real, X_clinical, gene_cols):
    logger.info("Generating synthetic data and constructing SCM Graphs...")
    device = torch.device(CONFIG['DEVICE'])
    netG.eval()
    
    n_synth = len(X_real)
    z = torch.randn(n_synth, CONFIG['Z_DIM']).to(device)
    y_syn = torch.randint(0, 2, (n_synth, 1)).float().to(device)
    
    with torch.no_grad():
        x_syn = netG(z, y_syn).cpu().numpy().reshape(n_synth, -1)
    y_syn = y_syn.cpu().numpy().flatten()
    
    # Combine Data
    X_real_flat = X_real.reshape(len(X_real), -1)
    X_comb = np.vstack([X_real_flat, x_syn]) # (2N, Genes)
    Y_comb = np.concatenate([Y_real, y_syn])
    
    clin_idx = np.random.choice(len(X_clinical), n_synth, replace=True)
    X_clin_syn = X_clinical[clin_idx]
    X_clin_comb = np.vstack([X_clinical, X_clin_syn]) # (2N, Clinical)
    
    # Build Graphs (The SCM part)
    # We build graphs on the combined dataset to capture relationships across real/syn samples
    # Note: X_comb contains gene expressions. X_clin_comb contains clinical info.
    # We use these to build the structure.
    adj_matrix = build_adjacency_matrices(X_clin_comb, X_comb)
    logger.info(f"Constructed Adjacency Tensor: {adj_matrix.shape}")
    
    # Candidates
    if os.path.exists(CONFIG['CANDIDATE_PATH']):
        cand_df = pd.read_csv(CONFIG['CANDIDATE_PATH'], sep="\t")
        if 'snp' in cand_df.columns:
            candidates = cand_df['snp'].tolist()
        else:
            candidates = cand_df.iloc[:, 0].tolist()
        candidates = [c for c in candidates if c in gene_cols]
    else:
        candidates = gene_cols[:20]

    results = []
    gene_col_map = {name: i for i, name in enumerate(gene_cols)}
    
    logger.info(f"Evaluating {len(candidates)} genes with GTN-DragonNet...")
    
    for gene in candidates:
        if gene not in gene_col_map: continue
        idx = gene_col_map[gene]
        
        # T: High/Low Expression
        gene_vals = X_comb[:, idx]
        T = (gene_vals > np.median(gene_vals)).astype(int)
        
        # W: Covariates for adjustment (Clinical)
        W = X_clin_comb
        Y = Y_comb
        
        try:
            # Train GTN-DragonNet
            model = train_gtn_dragonnet(W, T, Y, adj_matrix, epochs=50, device=CONFIG['DEVICE'])
            
            # Predict ATE
            model.eval()
            with torch.no_grad():
                W_t = torch.tensor(W, dtype=torch.float32).to(CONFIG['DEVICE'])
                A_t = adj_matrix.to(CONFIG['DEVICE'])
                y0_pred, y1_pred, t_pred, _ = model(A_t, W_t)
                ate = (y1_pred - y0_pred).mean().item()
            
            results.append({'SNP_Site': gene, 'ATE': ate, 'Abs_ATE': abs(ate)})
            
        except Exception as e:
            logger.error(f"GTN training failed for {gene}: {e}")
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values('Abs_ATE', ascending=False)
        out_path = os.path.join(CONFIG['OUTPUT_DIR'], 'gtn_rs_cgan_ate_results.csv')
        res_df.to_csv(out_path, index=False)
        logger.info(f"Saved GTN ATE results to {out_path}")
        print(res_df.head(15))

def main():
    if not os.path.exists(CONFIG['OUTPUT_DIR']):
        os.makedirs(CONFIG['OUTPUT_DIR'])
    
    X, Y, X_clin, gene_cols, df = load_and_preprocess()
    if X is None: return
    
    # 1. Train RS-CGAN (Feature Generator)
    netG = train_rs_cgan(X, Y)
    
    # 2. Estimate ATE with GTN-enhanced SCM
    estimate_ate_with_gtn(netG, X, Y, X_clin, gene_cols)

if __name__ == "__main__":
    main()
