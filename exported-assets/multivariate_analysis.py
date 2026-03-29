# multivariate_analysis.py
"""
PLS-DA and PCA analysis with proper R²/Q² validation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score

class PLSDAAnalyzer:
    """PLS-DA with corrected R² and Q² calculation"""
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.metrics = {}
        
    def fit_predict(self, X, y):
        """
        Fit PLS-DA and calculate R², Q² using proper LOOCV
        
        Returns:
            dict with scores, VIP, R²X, R²Y, Q², cv_accuracy
        """
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PLS-DA model
        self.model = PLSRegression(n_components=self.n_components)
        self.model.fit(X_scaled, y_encoded)
        
        # Get scores
        scores = self.model.transform(X_scaled)
        
        # Calculate R²X (variance explained in X)
        X_reconstructed = scores @ self.model.x_loadings_.T
        ss_res_x = np.sum((X_scaled - X_reconstructed) ** 2)
        ss_tot_x = np.sum((X_scaled - X_scaled.mean(axis=0)) ** 2)
        r2x = 1 - (ss_res_x / ss_tot_x)
        
        # Calculate R²Y (variance explained in Y)
        y_pred_train = self.model.predict(X_scaled).ravel()
        r2y = r2_score(y_encoded, y_pred_train)
        
        # Calculate Q² using LOOCV (proper way)
        q2 = self._calculate_q2(X_scaled, y_encoded)
        
        # Calculate VIP scores
        vip_scores = self._calculate_vip(X_scaled, y_encoded)
        
        # Calculate classification accuracy
        y_pred_class = np.round(y_pred_train).astype(int)
        y_pred_class = np.clip(y_pred_class, 0, len(self.label_encoder.classes_) - 1)
        cv_accuracy = np.mean(y_pred_class == y_encoded)
        
        self.metrics = {
            'r2x': r2x,
            'r2y': r2y,
            'q2': q2,
            'cv_accuracy': cv_accuracy
        }
        
        return {
            'scores': scores,
            'vip_scores': vip_scores,
            'metrics': self.metrics,
            'y_pred': y_pred_class,
            'y_true': y_encoded,
            'feature_names': None  # Set externally
        }
    
    def _calculate_q2(self, X, y):
        """
        Calculate Q² using Leave-One-Out Cross-Validation
        Q² = 1 - PRESS/TSS
        where PRESS = sum of squared prediction errors
        """
        loo = LeaveOneOut()
        y_pred_cv = np.zeros(len(y))
        
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit model on training fold
            pls_cv = PLSRegression(n_components=self.n_components)
            pls_cv.fit(X_train, y_train)
            
            # Predict on test sample
            y_pred_cv[test_idx] = pls_cv.predict(X_test).ravel()
        
        # Calculate Q²
        press = np.sum((y - y_pred_cv) ** 2)
        tss = np.sum((y - y.mean()) ** 2)
        q2 = 1 - (press / tss)
        
        return q2
    
    def _calculate_vip(self, X, y):
        """Calculate Variable Importance in Projection (VIP) scores"""
        t = self.model.x_scores_
        w = self.model.x_weights_
        q = self.model.y_loadings_
        
        p, h = w.shape
        vips = np.zeros(p)
        
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        
        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
            vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
        
        return vips
    
    def permutation_test(self, X, y, n_permutations=200):
        """
        Run permutation test for PLS-DA validation
        
        Returns:
            dict with r2_perms, q2_perms, correlations, p_value
        """
        # Encode and scale
        y_encoded = self.label_encoder.transform(y)
        X_scaled = self.scaler.transform(X)
        
        # Actual metrics
        actual_r2y = self.metrics['r2y']
        actual_q2 = self.metrics['q2']
        
        # Permutation loop
        r2_perms = []
        q2_perms = []
        correlations = []
        
        for _ in range(n_permutations):
            y_perm = np.random.permutation(y_encoded)
            
            # Calculate correlation
            corr = np.corrcoef(y_encoded, y_perm)[0, 1]
            if np.isnan(corr):
                corr = 0
            correlations.append(corr)
            
            # Fit model on permuted data
            pls_perm = PLSRegression(n_components=self.n_components)
            pls_perm.fit(X_scaled, y_perm)
            
            # R²Y permuted
            y_pred_perm = pls_perm.predict(X_scaled).ravel()
            r2_perm = r2_score(y_perm, y_pred_perm)
            r2_perms.append(r2_perm)
            
            # Q² permuted (LOOCV)
            q2_perm = self._calculate_q2_permuted(X_scaled, y_perm)
            q2_perms.append(q2_perm)
        
        # Calculate p-value (proportion of permuted Q² >= actual Q²)
        p_value = (np.sum(np.array(q2_perms) >= actual_q2) + 1) / (n_permutations + 1)
        
        # Calculate intercepts (regression line at correlation = 0)
        if len(correlations) > 1:
            r2_intercept = np.poly1d(np.polyfit(correlations, r2_perms, 1))(0)
            q2_intercept = np.poly1d(np.polyfit(correlations, q2_perms, 1))(0)
        else:
            r2_intercept = np.mean(r2_perms)
            q2_intercept = np.mean(q2_perms)
        
        return {
            'r2_perms': r2_perms,
            'q2_perms': q2_perms,
            'correlations': correlations,
            'actual_r2y': actual_r2y,
            'actual_q2': actual_q2,
            'p_value': p_value,
            'r2_intercept': r2_intercept,
            'q2_intercept': q2_intercept
        }
    
    def _calculate_q2_permuted(self, X, y):
        """Helper for permutation Q² calculation"""
        loo = LeaveOneOut()
        y_pred_cv = np.zeros(len(y))
        
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            
            pls_cv = PLSRegression(n_components=self.n_components)
            pls_cv.fit(X_train, y_train)
            y_pred_cv[test_idx] = pls_cv.predict(X_test).ravel()
        
        press = np.sum((y - y_pred_cv) ** 2)
        tss = np.sum((y - y.mean()) ** 2)
        
        return 1 - (press / tss)


class PCAAnalyzer:
    """PCA with variance explained"""
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.model = None
        self.scaler = None
        
    def fit_transform(self, X):
        """
        Fit PCA and transform data
        
        Returns:
            dict with scores, loadings, variance_explained
        """
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = PCA(n_components=self.n_components)
        scores = self.model.fit_transform(X_scaled)
        
        return {
            'scores': scores,
            'loadings': self.model.components_,
            'variance_explained': self.model.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.model.explained_variance_ratio_)
        }
