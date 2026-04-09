
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GAN_EVAL")

class GANEvaluator:
    """
    GAN 模型评估工具类，包含 FID, MMD, MSE 等指标。
    专为 SNP/组学数据设计，但也兼容图像格式。
    """
    
    def __init__(self, device='cpu'):
        self.device = device

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        计算 Fréchet Distance (FID 的核心公式)
        :param mu1: 真实数据的均值
        :param sigma1: 真实数据的协方差矩阵
        :param mu2: 生成数据的均值
        :param sigma2: 生成数据的协方差矩阵
        :return: Fréchet Distance
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            logger.warning("fid calculation produces singular product; adding %s to diagonal of cov estimates", eps)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            
        # Numerical error might give slight complex component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real
            
        tr_covmean = np.trace(covmean)
        
        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    def calculate_fid(self, real_data, fake_data, feature_extractor=None):
        """
        计算 FID (Fréchet Inception Distance)
        如果不提供 feature_extractor，则直接在原始数据空间（Flatten后）计算 FD。
        
        :param real_data: 真实数据 (N, D) 或 (N, C, H, W)
        :param fake_data: 生成数据 (N, D) 或 (N, C, H, W)
        :param feature_extractor: PyTorch 模型，用于提取特征。如果为 None，则使用原始数据。
        :return: FID Score
        """
        real_data = self._to_numpy(real_data)
        fake_data = self._to_numpy(fake_data)
        
        if feature_extractor is not None:
            # TODO: 实现基于特征提取器的特征计算
            # 目前简化为直接计算，假设输入已经是特征或不需要提取
            pass
        
        # Flatten if needed
        if len(real_data.shape) > 2:
            real_data = real_data.reshape(real_data.shape[0], -1)
            fake_data = fake_data.reshape(fake_data.shape[0], -1)
            
        # Calculate statistics
        mu1, sigma1 = np.mean(real_data, axis=0), np.cov(real_data, rowvar=False)
        mu2, sigma2 = np.mean(fake_data, axis=0), np.cov(fake_data, rowvar=False)
        
        fid = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid

    def calculate_mmd(self, real_data, fake_data, kernel='rbf', gamma=None):
        """
        计算 MMD (Maximum Mean Discrepancy)
        :param real_data: 真实数据
        :param fake_data: 生成数据
        :param kernel: 'rbf', 'polynomial', 'linear'
        :param gamma: Kernel parameter (for RBF/Poly)
        :return: MMD Score
        """
        real_data = self._to_numpy(real_data)
        fake_data = self._to_numpy(fake_data)
        
        # Flatten
        if len(real_data.shape) > 2:
            real_data = real_data.reshape(real_data.shape[0], -1)
            fake_data = fake_data.reshape(fake_data.shape[0], -1)
            
        # Subsample if data is too large for kernel matrix (e.g., > 2000 samples)
        # MMD is O(N^2)
        if real_data.shape[0] > 2000:
            idx = np.random.choice(real_data.shape[0], 2000, replace=False)
            real_data = real_data[idx]
        if fake_data.shape[0] > 2000:
            idx = np.random.choice(fake_data.shape[0], 2000, replace=False)
            fake_data = fake_data[idx]
            
        X = real_data
        Y = fake_data
        
        if kernel == 'rbf':
            XX = rbf_kernel(X, X, gamma=gamma)
            YY = rbf_kernel(Y, Y, gamma=gamma)
            XY = rbf_kernel(X, Y, gamma=gamma)
        elif kernel == 'linear':
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
            
        mmd = XX.mean() + YY.mean() - 2 * XY.mean()
        return mmd

    def calculate_mse(self, real_data, fake_data):
        """
        计算均方误差 (MSE) - 仅当数据对齐或比较统计量时有意义
        这里计算的是特征均值的 MSE，作为分布的一阶矩差异。
        """
        real_data = self._to_numpy(real_data)
        fake_data = self._to_numpy(fake_data)
        
        mu1 = np.mean(real_data, axis=0)
        mu2 = np.mean(fake_data, axis=0)
        
        mse = np.mean((mu1 - mu2) ** 2)
        return mse

    def _to_numpy(self, data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.array(data)

def evaluate_all_metrics(real_data, fake_data, device='cpu'):
    """
    一键评估所有指标
    """
    evaluator = GANEvaluator(device=device)
    
    logger.info("Calculating MSE (First Moment)...")
    mse = evaluator.calculate_mse(real_data, fake_data)
    
    logger.info("Calculating FID (Fréchet Distance)...")
    # 注意：高维数据计算 FID 可能较慢且数值不稳定，建议降维或使用部分特征
    # 这里为了演示直接计算，实际使用建议先 PCA 降维到 50-100 维
    fid = evaluator.calculate_fid(real_data, fake_data)
    
    logger.info("Calculating MMD (RBF Kernel)...")
    mmd = evaluator.calculate_mmd(real_data, fake_data, kernel='rbf')
    
    logger.info("Evaluation Results:")
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"FID: {fid:.6f}")
    logger.info(f"MMD: {mmd:.6f}")
    
    return {
        'MSE': mse,
        'FID': fid,
        'MMD': mmd
    }

if __name__ == "__main__":
    # Test run
    logger.info("Running test evaluation...")
    N, D = 100, 50
    real = np.random.randn(N, D)
    fake = np.random.randn(N, D) + 0.1 # Slight shift
    
    metrics = evaluate_all_metrics(real, fake)
    print(metrics)
