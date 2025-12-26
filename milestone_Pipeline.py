# milestone3_pipeline.py
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# =============================
# Threshold-Based Detector
# =============================
class ThresholdAnomalyDetector:
    def __init__(self):
        self.threshold_rules = {
            'heart_rate': {
                'metric_name': 'heart_rate',
                'min_threshold': 40,
                'max_threshold': 120,
                'sustained_minutes': 10,
                'description': 'HR outside normal range'
            },
            'steps': {
                'metric_name': 'step_count',
                'min_threshold': 0,
                'max_threshold': 1000,
                'sustained_minutes': 5,
                'description': 'Unrealistic steps detected'
            },
            'sleep': {
                'metric_name': 'duration_minutes',
                'min_threshold': 180,
                'max_threshold': 720,
                'sustained_minutes': 0,
                'description': 'Unusual sleep duration'
            }
        }
        self.detected_anomalies = []

    def detect(self, df: pd.DataFrame, data_type: str):
        report = {
            'method': 'threshold',
            'data_type': data_type,
            'total_rows': len(df),
            'anomalies_detected': 0,
            'percentage': 0
        }

        if data_type not in self.threshold_rules:
            return df, report

        rule = self.threshold_rules[data_type]
        metric = rule['metric_name']
        df_out = df.copy()
        df_out['threshold_anomaly'] = False

        too_high = df_out[metric] > rule['max_threshold']
        too_low = df_out[metric] < rule['min_threshold']

        if rule['sustained_minutes'] > 0:
            win = rule['sustained_minutes']
            high_sust = too_high.rolling(win).sum() >= win
            low_sust = too_low.rolling(win).sum() >= win
            df_out.loc[high_sust, 'threshold_anomaly'] = True
            df_out.loc[low_sust, 'threshold_anomaly'] = True
        else:
            df_out.loc[too_high | too_low, 'threshold_anomaly'] = True

        count = df_out['threshold_anomaly'].sum()
        report['anomalies_detected'] = int(count)
        report['percentage'] = (count / len(df_out)) * 100 if len(df_out) else 0

        return df_out, report


# =============================
# Prophet Residual Detector
# =============================
class ResidualAnomalyDetector:
    def __init__(self, threshold_std=3.0):
        self.threshold_std = threshold_std
        self.detected_anomalies = []

    def detect_from_prophet(self, df: pd.DataFrame, forecast: pd.DataFrame, data_type: str):
        report = {
            'method': 'prophet_residual',
            'data_type': data_type,
            'std': self.threshold_std,
            'anomalies_detected': 0
        }

        metric_map = {'heart_rate': 'heart_rate', 'steps': 'step_count', 'sleep': 'duration_minutes'}
        metric = metric_map.get(data_type)

        df_res = df.copy()
        f = forecast.copy().rename(columns={'ds': 'timestamp', 'yhat': 'predicted'})
        df_res = df_res.merge(f[['timestamp', 'predicted', 'yhat_lower', 'yhat_upper']], on='timestamp', how='left')

        df_res['residual'] = df_res[metric] - df_res['predicted']
        std = df_res['residual'].std()
        thr = self.threshold_std * std
        df_res['residual_anomaly'] = np.abs(df_res['residual']) > thr

        count = df_res['residual_anomaly'].sum()
        report['anomalies_detected'] = int(count)

        return df_res, report


# =============================
# Cluster-Based Detector
# =============================
class ClusterAnomalyDetector:
    def __init__(self):
        self.detected_anomalies = []

    def detect(self, features: pd.DataFrame, labels: np.ndarray, data_type: str, outlier_threshold=0.05):
        report = {
            'method': 'cluster',
            'data_type': data_type,
            'anomalies_detected': 0
        }

        df = features.copy()
        df['cluster'] = labels

        total = len(labels)
        small_clusters = []
        for lab, count in pd.Series(labels).value_counts().items():
            if lab == -1 or (count / total) < outlier_threshold:
                small_clusters.append(lab)

        df['cluster_anomaly'] = df['cluster'].isin(small_clusters)
        report['anomalies_detected'] = int(df['cluster_anomaly'].sum())

        return df, report


# =============================
# Visualization Helper
# =============================
class AnomalyVisualizer:
    def summarize(self, all_reports: Dict):
        total = 0
        breakdown = []
        for dt, methods in all_reports.items():
            for m, rep in methods.items():
                total += rep.get('anomalies_detected', 0)
                breakdown.append(rep)
        return total, breakdown


# =============================
# Complete Pipeline Controller
# =============================
class AnomalyDetectionPipeline:
    def __init__(self):
        self.threshold = ThresholdAnomalyDetector()
        self.residual = ResidualAnomalyDetector()
        self.cluster = ClusterAnomalyDetector()
        self.visualizer = AnomalyVisualizer()
        self.final_results = {}

    def run_complete_milestone3(self, preprocessed_data: Dict[str, pd.DataFrame],
                                prophet_forecasts: Optional[Dict] = None,
                                cluster_labels: Optional[Dict] = None,
                                feature_matrices: Optional[Dict] = None):

        output = {'data': {}, 'reports': {}, 'summary': {}}

        for d, df in preprocessed_data.items():
            output['reports'][d] = {}

            # 1️⃣ threshold
            tdf, tr = self.threshold.detect(df, d)
            output['reports'][d]['threshold'] = tr

            df_final = tdf

            # 2️⃣ prophet
            if prophet_forecasts and d in prophet_forecasts:
                p = prophet_forecasts[d]
                pdf, pr = self.residual.detect_from_prophet(df_final, p, d)
                df_final = pdf
                output['reports'][d]['residual'] = pr

            # 3️⃣ clustering
            if cluster_labels and d in cluster_labels and feature_matrices:
                lbl = cluster_labels[d]
                feat = feature_matrices[d]
                _, cr = self.cluster.detect(feat, lbl, d)
                output['reports'][d]['cluster'] = cr

            output['data'][d] = df_final

        total, breakdown = self.visualizer.summarize(output["reports"])
        output['summary'] = {'total_anomalies': total, 'methods': breakdown}

        self.final_results = output
        return output
