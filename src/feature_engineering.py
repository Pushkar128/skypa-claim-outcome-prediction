import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PayerDenialRateTransformer(BaseEstimator, TransformerMixin):
    """
    Computes payer denial rate strictly from the training dataset to prevent data leakage.
    payer_denial_rate = (Denied claims for payer in TRAIN) / (Total claims for payer in TRAIN)
    """
    def __init__(self, target_col='status', denied_label='Denied'):
        self.target_col = target_col
        self.denied_label = denied_label
        self.payer_denial_rates_ = {}
        self.global_denial_rate_ = 0.0

    def fit(self, X, y=None):
        X_copy = X.copy()
        if y is not None:
            X_copy[self.target_col] = y
            
        total_claims = len(X_copy)
        if total_claims > 0 and self.target_col in X_copy.columns:
            denied_count = (X_copy[self.target_col] == self.denied_label).sum()
            self.global_denial_rate_ = denied_count / total_claims
            
            payer_stats = X_copy.groupby('payer')[self.target_col].agg(
                total='count',
                denied=lambda s: (s == self.denied_label).sum()
            )
            
            for payer, row in payer_stats.iterrows():
                self.payer_denial_rates_[payer] = row['denied'] / row['total'] if row['total'] > 0 else self.global_denial_rate_
                
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out['payer_denial_rate'] = X_out['payer'].map(self.payer_denial_rates_).fillna(self.global_denial_rate_)
        return X_out


class CptAvgBilledRatioTransformer(BaseEstimator, TransformerMixin):
    """
    Computes the ratio of billed_amount to the mean billed_amount for that CPT code in TRAIN data.
    billed_to_cpt_ratio = billed_amount / mean_billed_amount_for_cpt_in_train
    """
    def __init__(self):
        self.cpt_avg_billed_ = {}
        self.global_avg_billed_ = 200.0

    def fit(self, X, y=None):
        X_copy = X.copy()
        if 'billed_amount' in X_copy.columns and 'cpt_code' in X_copy.columns:
            self.global_avg_billed_ = float(X_copy['billed_amount'].mean())
            grouped = X_copy.groupby('cpt_code')['billed_amount'].mean()
            self.cpt_avg_billed_ = grouped.to_dict()
        return self

    def transform(self, X):
        X_out = X.copy()
        cpt_avg_series = X_out['cpt_code'].map(self.cpt_avg_billed_).fillna(self.global_avg_billed_)
        X_out['billed_to_cpt_ratio'] = X_out['billed_amount'] / (cpt_avg_series + 1e-5)
        return X_out
