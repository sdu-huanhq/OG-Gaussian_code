import torch
import clip
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score
from typing import List


class CLIPTextEncoder:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, preprocess = clip.load(model_name, device=self.device)


    def encode_text(self, text: List[str], reduce_dim=None, random_state=42, whiten=True) -> np.ndarray:
        text = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(text).cpu().numpy()
        
        if reduce_dim is not None:
            if reduce_dim > min(features.shape[0], features.shape[1]):
                return None
            scaler = StandardScaler()
            std_feat = scaler.fit_transform(features)
            pca = PCA(n_components=reduce_dim, random_state=random_state, whiten=whiten)
            pca_feat = pca.fit_transform(std_feat)

            pca_feat_recon = pca.inverse_transform(pca_feat)
            mse_pca = mean_squared_error(features, pca_feat_recon)
            evr_pca = explained_variance_score(features, pca_feat_recon)
            
            features = pca_feat
        
        return features
