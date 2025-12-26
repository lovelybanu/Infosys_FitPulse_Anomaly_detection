# pages/2_Milestone_2.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Try optional heavy libraries, otherwise provide fallbacks
try:
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction.settings import MinimalFCParameters, ComprehensiveFCParameters
    TSFRESH_AVAILABLE = True
except Exception:
    TSFRESH_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -------------------------
# Utility functions
# -------------------------
def ensure_timestamp(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    if ts_col not in df.columns:
        raise ValueError("timestamp column missing")
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    return df

def basic_features_from_series(series: pd.Series) -> Dict[str, float]:
    """Compute a small set of statistical features for a numeric series"""
    s = series.dropna()
    f = {}
    if len(s) == 0:
        return {
            "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
            "skew": np.nan, "kurtosis": np.nan, "median": np.nan,
        }
    f["mean"] = float(s.mean())
    f["std"] = float(s.std())
    f["min"] = float(s.min())
    f["max"] = float(s.max())
    f["median"] = float(s.median())
    f["skew"] = float(s.skew())
    f["kurtosis"] = float(s.kurtosis())
    # simple autocorrelation lag-1
    if len(s) > 1:
        f["autocorr_lag1"] = float(s.autocorr(lag=1))
    else:
        f["autocorr_lag1"] = np.nan
    return f

def rmssd(series: pd.Series) -> float:
    """Root Mean Square of Successive Differences - HRV proxy"""
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    diffs = np.diff(s.values)
    return float(np.sqrt(np.mean(diffs ** 2)))

# -------------------------
# Feature extraction classes
# -------------------------
class FlexibleFeatureExtractor:
    """Extract features: tries TSFresh, else uses fast handcrafted features"""
    def __init__(self, complexity: str = "minimal"):
        self.complexity = complexity
        self.feature_matrix = pd.DataFrame()
        self.report = {}

    def extract(self, df: pd.DataFrame, data_type: str, window_size: int = 60) -> Tuple[pd.DataFrame, Dict]:
        """
        df: preprocessed dataframe with timestamp and metric column
        data_type: 'heart_rate'|'steps'|'sleep'
        window_size: integer number of rows (assuming resampled freq ~ 1min)
        """
        report = {"data_type": data_type, "rows": len(df), "window_size": window_size, "success": False}
        try:
            df = ensure_timestamp(df)
            metric_map = {"heart_rate": "heart_rate", "steps": "step_count", "sleep": "duration_minutes"}
            if data_type not in metric_map or metric_map[data_type] not in df.columns:
                report["error"] = "metric column missing"
                return pd.DataFrame(), report
            metric_col = metric_map[data_type]

            # If tsfresh available and user selected 'tsfresh' behavior, use it
            if TSFRESH_AVAILABLE:
                # Prepare rolling windows (50% overlap) - convert window_size minutes to rows (assume 1 row/min)
                step = max(1, window_size // 2)
                prepared = []
                window_id = 0
                for i in range(0, max(0, len(df) - window_size + 1), step):
                    w = df.iloc[i:i+window_size].copy()
                    w["window_id"] = window_id
                    prepared.append(w[["window_id", "timestamp", metric_col]])
                    window_id += 1
                if not prepared:
                    report["error"] = "not enough rows for windows"
                    return pd.DataFrame(), report
                df_prep = pd.concat(prepared, ignore_index=True)
                df_prep = df_prep.rename(columns={metric_col: "value"})
                # choose parameters
                if self.complexity == "minimal":
                    fc = MinimalFCParameters()
                else:
                    try:
                        fc = ComprehensiveFCParameters()
                    except Exception:
                        fc = MinimalFCParameters()
                feat = extract_features(df_prep, column_id="window_id", column_sort="timestamp",
                                        default_fc_parameters=fc, n_jobs=1)
                feat = impute(feat)
                # remove constant cols
                feat = feat.loc[:, feat.std() != 0]
                self.feature_matrix = feat
                report["features"] = feat.shape[1]
                report["windows"] = feat.shape[0]
                report["success"] = True
                self.report = report
                return feat, report

            # FALLBACK: handcrafted features
            # Create sliding windows and compute summary features
            step = max(1, window_size // 2)
            rows = []
            window_id = 0
            for i in range(0, max(0, len(df) - window_size + 1), step):
                w = df.iloc[i:i+window_size]
                s = w[metric_col]
                feats = basic_features_from_series(s)
                feats["rmssd"] = rmssd(s) if metric_col == "heart_rate" else np.nan
                feats["window_id"] = window_id
                feats["start_ts"] = w["timestamp"].iloc[0]
                rows.append(feats)
                window_id += 1
            if not rows:
                report["error"] = "not enough rows for handcrafted windows"
                return pd.DataFrame(), report
            feat_df = pd.DataFrame(rows).set_index("window_id")
            self.feature_matrix = feat_df
            report["features"] = feat_df.shape[1]
            report["windows"] = feat_df.shape[0]
            report["success"] = True
            self.report = report
            return feat_df, report

        except Exception as e:
            report["error"] = str(e)
            return pd.DataFrame(), report

# -------------------------
# Trend modeler (Prophet fallback)
# -------------------------
class TrendModeler:
    """Fit trend model using Prophet if present, else use rolling median + residuals"""
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.residuals = {}

    def fit_and_forecast(self, df: pd.DataFrame, data_type: str, metric_col: str, forecast_periods: int = 60) -> Tuple[pd.DataFrame, Dict]:
        report = {"data_type": data_type, "rows": len(df), "success": False}
        try:
            df = ensure_timestamp(df)
            df = df.dropna(subset=[metric_col])
            if len(df) < 2:
                report["error"] = "insufficient data"
                return pd.DataFrame(), report

            if PROPHET_AVAILABLE:
                # --- Ensure timestamps are timezone-naive for Prophet ---
                ts = pd.to_datetime(df["timestamp"], errors="coerce")

                # If timezone-aware, convert to UTC then remove tz info
                try:
                    if ts.dt.tz is not None:
                        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
                    else:
                        # in some corner cases, attempt to remove tz info safely
                        try:
                            ts = ts.dt.tz_localize(None)
                        except Exception:
                            pass
                except Exception:
                    # If dt accessor fails, coerce to naive
                    ts = pd.to_datetime(ts).dt.tz_localize(None)

                prophet_df = pd.DataFrame({"ds": ts, "y": df[metric_col].values})

                # Remove NaNs and ensure minimum rows
                prophet_df = prophet_df.dropna(subset=["ds", "y"]).reset_index(drop=True)
                if len(prophet_df) < 2:
                    report["error"] = "Insufficient valid ds/y after timezone cleaning"
                    return pd.DataFrame(), report

                model = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=forecast_periods, freq='min')
                forecast = model.predict(future)
                # residuals: merge original points
                merged = prophet_df.merge(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="left")
                merged["residual"] = merged["y"] - merged["yhat"]
                merged["residual_abs"] = merged["residual"].abs()
                self.models[data_type] = model
                self.forecasts[data_type] = forecast
                self.residuals[data_type] = merged
                report["success"] = True
                return forecast, report

            # FALLBACK simple trend: rolling median -> residuals
            window = 15
            df_sorted = df.sort_values("timestamp").copy().reset_index(drop=True)
            df_sorted["trend"] = df_sorted[metric_col].rolling(window=window, min_periods=1, center=True).median()
            df_sorted["residual"] = df_sorted[metric_col] - df_sorted["trend"]
            df_sorted["residual_abs"] = df_sorted["residual"].abs()
            # make a simple forecast by extending last trend value
            last_ts = df_sorted["timestamp"].max()
            freq = pd.infer_freq(df_sorted["timestamp"].dropna()) or "T"
            future_index = pd.date_range(start=last_ts + pd.Timedelta(1, unit="m"), periods=forecast_periods, freq=freq)
            forecast = pd.DataFrame({"ds": np.concatenate([df_sorted["timestamp"].values, future_index.values])})
            forecast["yhat"] = np.concatenate([df_sorted["trend"].values, np.repeat(df_sorted["trend"].iloc[-1], forecast_periods)])
            forecast["yhat_lower"] = forecast["yhat"] - df_sorted["residual_abs"].std()
            forecast["yhat_upper"] = forecast["yhat"] + df_sorted["residual_abs"].std()
            merged = df_sorted.rename(columns={"timestamp": "ds", metric_col: "y"})
            merged = merged.merge(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="left")
            if "y" in merged:
                merged["residual"] = merged["y"] - merged["yhat"]
                merged["residual_abs"] = merged["residual"].abs()
            self.forecasts[data_type] = forecast
            self.residuals[data_type] = merged
            report["success"] = True
            return forecast, report
        except Exception as e:
            report["error"] = str(e)
            return pd.DataFrame(), report

    def anomalies_from_residuals(self, data_type: str, threshold_std: float = 3.0) -> pd.DataFrame:
        if data_type not in self.residuals:
            return pd.DataFrame()
        res = self.residuals[data_type].dropna(subset=["residual"])
        mean_r = res["residual"].mean()
        std_r = res["residual"].std()
        if pd.isna(std_r) or std_r == 0:
            return pd.DataFrame()
        thr = threshold_std * std_r
        anomalies = res[(res["residual"] > mean_r + thr) | (res["residual"] < mean_r - thr)].copy()
        anomalies["anomaly_score"] = (anomalies["residual"] - mean_r).abs() / std_r
        return anomalies

    # optional ensemble combining residual+IsolationForest (if sklearn available)
    def ensemble_anomalies(self, data_type: str, residual_thresh_std: float = 3.0) -> pd.DataFrame:
        res = self.residuals.get(data_type, pd.DataFrame()).dropna(subset=["residual"])
        if res.empty:
            return pd.DataFrame()
        mean_r = res["residual"].mean(); std_r = res["residual"].std()
        thr = residual_thresh_std * std_r
        res["anomaly_residual"] = ((res["residual"] > mean_r + thr) | (res["residual"] < mean_r - thr)).astype(int)
        if SKLEARN_AVAILABLE:
            try:
                iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
                preds = iso.fit_predict(res[["residual"]].fillna(0).values)
                res["anomaly_iso"] = (preds == -1).astype(int)
            except Exception:
                res["anomaly_iso"] = 0
        else:
            res["anomaly_iso"] = 0
        res["anomaly_score"] = res["anomaly_residual"] + res["anomaly_iso"]
        return res[res["anomaly_score"] > 0].copy()

# -------------------------
# Clusterer (KMeans + DBSCAN)
# -------------------------
class Clusterer:
    """
    Cluster feature matrices using KMeans or DBSCAN.
    Falls back to random labels if sklearn is missing.
    """
    def __init__(self):
        self.scalers = {}
        self.kmeans_models = {}
        self.dbscan_models = {}
        self.labels = {}
        self.reports = {}
        self.reduced = {}

    def run(self, feature_matrix: pd.DataFrame, method: str = "kmeans", n_clusters: int = 3,
            eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, Dict]:
        """
        method: 'kmeans' or 'dbscan'
        """
        report = {
            "method": method,
            "n_samples": len(feature_matrix),
            "n_features": feature_matrix.shape[1] if not feature_matrix.empty else 0,
            "success": False,
        }
        if feature_matrix.empty:
            report["error"] = "empty feature matrix"
            return np.array([]), report

        X = feature_matrix.copy().fillna(0).values

        if not SKLEARN_AVAILABLE:
            # fallback random labels
            labels = np.random.randint(0, n_clusters, size=X.shape[0])
            report["warning"] = "sklearn not available; random labels assigned"
            report["success"] = True
            self.labels["fallback"] = labels
            self.reports["fallback"] = report
            return labels, report

        # Standardize
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        self.scalers["last"] = scaler

        try:
            if method == "kmeans":
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(Xs)
                self.kmeans_models["last"] = model
                report["inertia"] = getattr(model, "inertia_", None)
            elif method == "dbscan":
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(Xs)
                self.dbscan_models["last"] = model
                # DBSCAN labels: -1 = noise
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                report["n_clusters_found"] = int(n_clusters_found)
                report["n_noise_points"] = int((labels == -1).sum())
            else:
                report["error"] = f"Unknown method: {method}"
                return np.array([]), report

            # clustering metrics (when >1 real cluster)
            unique_labels = np.unique(labels)
            report["n_unique_labels"] = int(len(unique_labels))
            if len(unique_labels) > 1 and len(labels) > len(unique_labels):
                try:
                    report["silhouette"] = silhouette_score(Xs, labels)
                except Exception:
                    pass
                try:
                    report["davies_bouldin"] = davies_bouldin_score(Xs, labels)
                except Exception:
                    pass

            report["success"] = True
            self.labels["last"] = labels
            self.reports["last"] = report
            return labels, report

        except Exception as e:
            report["error"] = str(e)
            return np.array([]), report

    def pca_and_plot(self, feature_matrix: pd.DataFrame, labels: np.ndarray, title_suffix: str = ""):
        """Project with PCA and plot colored scatter."""
        if feature_matrix.empty or labels is None or len(labels) == 0:
            st.warning("No features or labels to visualize")
            return

        if not SKLEARN_AVAILABLE:
            st.info("sklearn is not available — skipping PCA visualization")
            return

        scaler = StandardScaler()
        Xs = scaler.fit_transform(feature_matrix.fillna(0).values)
        pca = PCA(n_components=2, random_state=42)
        comps = pca.fit_transform(Xs)
        df_viz = pd.DataFrame({"pc1": comps[:, 0], "pc2": comps[:, 1], "cluster": labels.astype(str)})
        fig = px.scatter(df_viz, x="pc1", y="pc2", color="cluster",
                         title=f"Cluster projection (PCA) {title_suffix}")
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        st.plotly_chart(fig, use_container_width=True)

        # statistics
        unique, counts = np.unique(labels, return_counts=True)
        stats = pd.DataFrame({"cluster": unique, "size": counts})
        st.subheader("Cluster statistics")
        st.dataframe(stats, use_container_width=True)

# -------------------------
# Milestone 2 Page
# -------------------------
def main():
    st.set_page_config(page_title="Milestone 2 - Features & Modeling", page_icon="🔬", layout="wide")
    st.title("🔬Feature Extraction & Modeling")
    st.markdown("TSFresh (when available) ➜ Trend modeling (Prophet if available) ➜ Clustering (KMeans/DBSCAN)")

    # Sidebar controls
    st.sidebar.header("Settings")
    use_sample = st.sidebar.checkbox("Use sample data", value=False)
    window_size = st.sidebar.slider("Window size (rows/minutes)", min_value=10, max_value=240, value=60, step=10)
    forecast_periods = st.sidebar.slider("Forecast horizon (minutes)", min_value=30, max_value=720, value=120, step=30)
    clustering_method = st.sidebar.selectbox("Clustering method", ["kmeans", "dbscan"], index=0)
    n_clusters = st.sidebar.slider("KMeans clusters", min_value=2, max_value=8, value=3)

    # prepare data source: from Milestone 1 if available
    processed_data = {}
    if not use_sample and "preprocessor" in st.session_state and getattr(st.session_state.preprocessor, "processed_data", None):
        processed_data = st.session_state.preprocessor.processed_data
        st.success("Using processed data from Previous Step")
    else:
        st.info("Using sample generated data for Previous Step")
        # generate small sample time-series (1-min freq)
        idx = pd.date_range("2024-01-15 08:00:00", periods=360, freq="1T")
        base = 70 + 5*np.sin(np.linspace(0, 6.28, len(idx)))
        hr = (base + np.random.normal(0,2,len(idx))).clip(45,150)
        processed_data = {"heart_rate": pd.DataFrame({"timestamp": idx, "heart_rate": hr})}

    # initialize pipeline objects
    extractor = FlexibleFeatureExtractor(complexity="minimal")
    trend = TrendModeler()
    if "clusterer" not in st.session_state:
        st.session_state.clusterer = Clusterer()
    clusterer = st.session_state.clusterer

    results = {"features": {}, "forecasts": {}, "clusters": {}, "reports": {}}

    st.markdown("---")
    # process each data stream
    for data_type, df in processed_data.items():
        st.header(f"▶ {data_type.replace('_',' ').title()}")
        st.write(f"Rows: {len(df)}")
        col1, col2 = st.columns([2,1])

        with col1:
            # Feature extraction
            with st.expander("Step 1 — Feature Extraction", expanded=True):
                feat_df, feat_report = extractor.extract(df, data_type, window_size)
                if not feat_df.empty:
                    st.success(f"Extracted {feat_report.get('features', feat_df.shape[1])} features across {feat_report.get('windows', feat_df.shape[0])} windows")
                    st.dataframe(feat_df.head(10))
                    results["features"][data_type] = feat_df
                    results["reports"][f"{data_type}_features"] = feat_report
                    # show top features by variance
                    variances = feat_df.var().sort_values(ascending=False).head(10).to_frame("variance")
                    st.subheader("Top features by variance")
                    st.dataframe(variances)
                    # plot distributions of top 4 features
                    top_feats = variances.index.tolist()[:4]
                    for fcol in top_feats:
                        fig = px.histogram(feat_df, x=fcol, nbins=30, title=fcol)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No features extracted: " + str(feat_report.get("error", "")))

        with col2:
            # Trend modeling
            with st.expander("Step 2 — Trend Modeling & Anomaly Detection", expanded=True):
                metric_map = {"heart_rate": "heart_rate", "steps": "step_count", "sleep": "duration_minutes"}
                metric_col = metric_map.get(data_type)
                if metric_col and metric_col in df.columns:
                    forecast, model_report = trend.fit_and_forecast(df, data_type, metric_col, forecast_periods)
                    if not forecast.empty:
                        results["forecasts"][data_type] = forecast
                        results["reports"][f"{data_type}_model"] = model_report
                        st.success("Modeling complete")
                        # plot forecast
                        try:
                            if PROPHET_AVAILABLE:
                                # Prophet forecast plot
                                fig = go.Figure()
                                orig = df.dropna(subset=[metric_col])
                                fig.add_trace(go.Scatter(x=orig["timestamp"], y=orig[metric_col], mode="markers", name="Actual", marker=dict(size=4)))
                                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast"))
                                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
                                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", fill='tonexty', fillcolor='rgba(255,0,0,0.1)', line=dict(width=0), showlegend=False))
                                fig.update_layout(height=350, title=f"Forecast ({data_type})")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # fallback simple forecast view
                                fig = px.line(forecast, x="ds", y="yhat", title=f"Fallback Forecast ({data_type})")
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning("Could not render forecast plot: " + str(e))
                        # anomalies from residuals
                        anomalies = trend.anomalies_from_residuals(data_type, threshold_std=3.0)
                        if not anomalies.empty:
                            st.warning(f"Prophet/fallback detected {len(anomalies)} anomalies (residual threshold).")
                            st.dataframe(anomalies.head(10))
                    else:
                        st.warning("Modeling failed: " + str(model_report.get("error", "")))
                else:
                    st.info("No metric column available for modeling in this data stream.")

    st.markdown("---")
    # -------------------------
    # Clustering (on extracted features)
    # -------------------------
    st.header("🔬 Step 3 — Clustering (Behavioral Patterns)")

    if results["features"]:
        data_types_available = list(results["features"].keys())
        chosen = st.selectbox("Choose feature matrix to cluster", options=data_types_available)
        feat = results["features"][chosen]

        if not feat.empty:
            with st.expander("Configure Clustering", expanded=True):
                method = st.selectbox("Method", ["kmeans", "dbscan"], index=0)
                if method == "kmeans":
                    k = st.slider("K (for KMeans)", min_value=2, max_value=12, value=n_clusters)
                    run_button = st.button("Run KMeans", key=f"cluster_kmeans_{chosen}")
                    if run_button:
                        labels, creport = clusterer.run(feat, method="kmeans", n_clusters=k)
                        if labels.size > 0:
                            results["clusters"][chosen] = labels
                            results["reports"][f"{chosen}_cluster"] = creport
                            st.success(f"KMeans complete. {creport.get('n_unique_labels', '---')} labels")
                            clusterer.pca_and_plot(feat, labels, title_suffix=f"(KMeans, k={k})")
                        else:
                            st.warning("KMeans did not return labels: " + str(creport.get("error", "")))
                else:
                    eps = st.slider("DBSCAN eps", min_value=0.1, max_value=5.0, value=0.5, step=0.05)
                    min_samples = st.slider("DBSCAN min_samples", min_value=1, max_value=20, value=5)
                    run_button = st.button("Run DBSCAN", key=f"cluster_dbscan_{chosen}")
                    if run_button:
                        labels, creport = clusterer.run(feat, method="dbscan", eps=eps, min_samples=min_samples)
                        if labels.size > 0:
                            results["clusters"][chosen] = labels
                            results["reports"][f"{chosen}_cluster"] = creport
                            st.success(f"DBSCAN complete. {creport.get('n_unique_labels', '---')} labels, noise={creport.get('n_noise_points',0)}")
                            clusterer.pca_and_plot(feat, labels, title_suffix=f"(DBSCAN, eps={eps}, min_samples={min_samples})")
                        else:
                            st.warning("DBSCAN did not return labels: " + str(creport.get("error", "")))
    else:
        st.info("No feature matrices available to cluster. Run feature extraction first.")

    st.markdown("---")
    st.header("Overall Summary")
    st.write("Feature matrices:", {k: v.shape for k,v in results["features"].items()})
    st.write("Forecasts:", {k: (v.shape[0] if hasattr(v,'shape') else "---") for k,v in results["forecasts"].items()})
    st.write("Clusters:", {k: (len(v) if hasattr(v,'__len__') else 0) for k,v in results["clusters"].items()})

    # persist results for downstream pages
    st.session_state.milestone2_results = results



if __name__ == "__main__":
    main()
