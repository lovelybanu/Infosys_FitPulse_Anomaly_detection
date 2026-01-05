import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


    

# ======================================================
# A. FILE UPLOAD
# ======================================================
class FitnessDataUploader:
    """Handles file upload and initial data loading for fitness tracker data"""

    def __init__(self):
        self.supported_formats = [".csv", ".json"]
        self.required_columns = {
            "heart_rate": ["timestamp", "heart_rate"],
            "sleep": ["timestamp", "sleep_stage", "duration_minutes"],
            "steps": ["timestamp", "step_count"],
        }

    def create_upload_interface(self) -> Dict[str, pd.DataFrame]:
        st.subheader("📁 Upload Fitness Tracker Data")

        # If data was previously uploaded and stored in session, offer to reuse or clear it
        existing = st.session_state.get("raw_fitness_data", None)
        col1, col2 = st.columns([8, 2])
        with col1:
            if existing:
                st.info("Using previously uploaded data (persisted in session).")
        with col2:
            if existing:
                if st.button("Clear upload", key="clear_uploaded"):
                    # Clear the stored data and trigger a rerun to show the uploader immediately
                    st.session_state["raw_fitness_data"] = None
                    st.rerun()

        # If there's existing parsed data and the user did not clear it, return it
        if existing:
            return existing

        uploaded_files = st.file_uploader(
            "Upload a combined CSV/JSON file",
            type=["csv", "json"],
            accept_multiple_files=False,
            key="fitness_uploader",
        )

        data_dict: Dict[str, pd.DataFrame] = {}

        if not uploaded_files:
            st.info("Upload one combined fitness dataset file.")
            return data_dict

        try:
            if uploaded_files.name.endswith(".csv"):
                df = pd.read_csv(uploaded_files)
            elif uploaded_files.name.endswith(".json"):
                df = pd.DataFrame(json.load(uploaded_files))
            else:
                st.error("Unsupported file type.")
                return data_dict

            df.columns = df.columns.str.lower().str.strip()

            # ✅ SPLIT DATA AUTOMATICALLY
            if "heart_rate" in df.columns:
                heart_df = df[["timestamp", "heart_rate"]].dropna()
                data_dict["heart_rate"] = heart_df
                st.success(f"✅ Heart Rate Data Loaded: {len(heart_df)} rows")

            if "step_count" in df.columns:
                steps_df = df[["timestamp", "step_count"]].dropna()
                data_dict["steps"] = steps_df
                st.success(f"✅ Steps Data Loaded: {len(steps_df)} rows")

            if "sleep_stage" in df.columns and "duration_minutes" in df.columns:
                sleep_df = df[["timestamp", "sleep_stage", "duration_minutes"]].dropna()
                data_dict["sleep"] = sleep_df
                st.success(f"✅ Sleep Data Loaded: {len(sleep_df)} rows")

            if not data_dict:
                st.error("❌ No valid fitness columns found in the uploaded file.")
                st.stop()
            # Persist parsed data in session state so it survives page navigation
            st.session_state["raw_fitness_data"] = data_dict
        except Exception as e:
            st.error(f"Error reading file: {e}")

        return data_dict



# ======================================================
# B. VALIDATION & CLEANING
# ======================================================
class FitnessDataValidator:
    """Validates and cleans fitness tracker data"""

    def __init__(self):
        self.validation_rules = {
            "heart_rate": {"min_value": 30, "max_value": 220, "data_type": "numeric"},
            "step_count": {"min_value": 0, "max_value": 100000, "data_type": "numeric"},
            "duration_minutes": {
                "min_value": 0,
                "max_value": 1440,
                "data_type": "numeric",
            },
        }

    def validate_and_clean_data(
        self, df: pd.DataFrame, data_type: str
    ) -> Tuple[pd.DataFrame, Dict]:
        validation_report: Dict = {
            "original_rows": len(df),
            "issues_found": [],
            "rows_removed": 0,
            "missing_values_handled": 0,
            "outliers_flagged": 0,
        }

        try:
            df_clean = df.copy()
            df_clean = self._standardize_columns(df_clean)
            df_clean, timestamp_issues = self._clean_timestamps(df_clean)
            validation_report["issues_found"].extend(timestamp_issues)

            df_clean, numeric_issues = self._validate_numeric_columns(
                df_clean, data_type
            )
            validation_report["issues_found"].extend(numeric_issues)

            df_clean, missing_count = self._handle_missing_values(df_clean, data_type)
            validation_report["missing_values_handled"] = missing_count

            df_clean, outlier_count = self._detect_outliers(df_clean, data_type)
            validation_report["outliers_flagged"] = outlier_count

            initial_len = len(df_clean)
            df_clean = self._remove_invalid_rows(df_clean)
            validation_report["rows_removed"] = initial_len - len(df_clean)
            validation_report["final_rows"] = len(df_clean)
            validation_report["success"] = True

        except Exception as e:
            validation_report["success"] = False
            validation_report["error"] = str(e)

        return df_clean, validation_report

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        column_mapping = {
            "time": "timestamp",
            "date": "timestamp",
            "datetime": "timestamp",
            "hr": "heart_rate",
            "heartrate": "heart_rate",
            "heart rate": "heart_rate",
            "steps": "step_count",
            "step": "step_count",
            "stepcount": "step_count",
            "sleep": "sleep_stage",
            "stage": "sleep_stage",
            "duration": "duration_minutes",
        }
        df_renamed = df.rename(columns=column_mapping)
        df_renamed.columns = df_renamed.columns.str.lower().str.replace(" ", "_")
        return df_renamed

    def _clean_timestamps(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        issues: List[str] = []
        if "timestamp" not in df.columns:
            issues.append("No `timestamp` column found")
            return df, issues

        try:
            parsed_timestamps = pd.to_datetime(
                df["timestamp"], errors="coerce", infer_datetime_format=True
            )
            failed_count = parsed_timestamps.isna().sum()
            if failed_count > 0:
                issues.append(f"Failed to parse {failed_count} timestamp values")

            df["timestamp"] = parsed_timestamps

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)


            # if df["timestamp"].dt.tz is not None:
            #     df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
            # else:
            #     df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

        except Exception as e:
            issues.append(f"Timestamp processing error: {str(e)}")

        return df, issues

    def _validate_numeric_columns(
        self, df: pd.DataFrame, data_type: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        issues: List[str] = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col in self.validation_rules:
                rule = self.validation_rules[col]
                min_val = rule["min_value"]
                max_val = rule["max_value"]
                below_min = (df[col] < min_val).sum()
                above_max = (df[col] > max_val).sum()
                if below_min > 0 or above_max > 0:
                    issues.append(
                        f"{col}: {below_min} values below {min_val}, "
                        f"{above_max} values above {max_val}. Values clipped."
                    )
                    df[col] = df[col].clip(lower=min_val, upper=max_val)
        return df, issues

    def _handle_missing_values(
        self, df: pd.DataFrame, data_type: str
    ) -> Tuple[pd.DataFrame, int]:
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            return df, 0

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if col == "timestamp":
                    df = df.dropna(subset=["timestamp"])
                elif col in ["heart_rate", "step_count"]:
                    df[col] = df[col].fillna(method="ffill", limit=5)
                    df[col] = df[col].interpolate(method="linear")
                elif col == "duration_minutes":
                    median_duration = df[col].median()
                    df[col] = df[col].fillna(median_duration)
                elif col == "sleep_stage":
                    df[col] = df[col].fillna(method="ffill")

        return df, missing_count

    def _detect_outliers(
        self, df: pd.DataFrame, data_type: str
    ) -> Tuple[pd.DataFrame, int]:
        outlier_count = 0
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col != "timestamp":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                df[f"{col}_outlier"] = outliers
                outlier_count += int(outliers.sum())

        return df, outlier_count

    def _remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns:
            df = df.dropna(subset=["timestamp"])
        value_columns = [
            col for col in df.columns if col != "timestamp" and not col.endswith("_outlier")
        ]
        if value_columns:
            df = df.dropna(subset=value_columns, how="all")
        return df

    def generate_validation_report(self, validation_report: Dict) -> str:
        report_lines = [
            "DATA VALIDATION REPORT",
            "======================",
            f"Original rows        : {validation_report['original_rows']}",
            f"Final rows           : {validation_report.get('final_rows', 'N/A')}",
            f"Rows removed         : {validation_report['rows_removed']}",
            f"Missing values fixed : {validation_report['missing_values_handled']}",
            f"Outliers flagged     : {validation_report['outliers_flagged']}",
            "",
            "Issues Found:",
        ]
        if validation_report["issues_found"]:
            for issue in validation_report["issues_found"]:
                report_lines.append(f"• {issue}")
        else:
            report_lines.append("• No major issues detected")
        return "\n".join(report_lines)


# ======================================================
# C. TIME ALIGNMENT
# ======================================================
class TimeAligner:
    """Handles time alignment and resampling of fitness data"""

    def __init__(self):
        self.supported_frequencies = {
            "1min": "1T",
            "5min": "5T",
            "15min": "15T",
            "30min": "30T",
            "1hour": "1H",
        }

    def align_and_resample(
        self,
        df: pd.DataFrame,
        data_type: str,
        target_frequency: str = "1min",
        fill_method: str = "interpolate",
    ) -> Tuple[pd.DataFrame, Dict]:

        alignment_report: Dict = {
            "original_frequency": None,
            "target_frequency": target_frequency,
            "original_rows": len(df),
            "resampled_rows": 0,
            "gaps_filled": 0,
            "method_used": fill_method,
            "success": False,
        }

        try:
            if "timestamp" not in df.columns:
                raise ValueError("No `timestamp` column found")

            df_indexed = df.set_index("timestamp").sort_index()
            alignment_report["original_frequency"] = self._detect_frequency(df_indexed)

            if target_frequency not in self.supported_frequencies:
                raise ValueError(f"Unsupported frequency: {target_frequency}")

            freq_str = self.supported_frequencies[target_frequency]
            df_resampled = self._resample_by_type(df_indexed, data_type, freq_str)
            df_filled, gaps_filled = self._fill_missing_after_resample(
                df_resampled, data_type, fill_method
            )
            df_final = df_filled.reset_index()

            alignment_report["resampled_rows"] = len(df_final)
            alignment_report["gaps_filled"] = gaps_filled
            alignment_report["success"] = True

            return df_final, alignment_report

        except Exception as e:
            alignment_report["error"] = str(e)
            return df, alignment_report

    def _detect_frequency(self, df_indexed: pd.DataFrame) -> str:
        try:
            if len(df_indexed) < 2:
                return "insufficient_data"

            time_diffs = df_indexed.index.to_series().diff().dropna()
            mode_diff = time_diffs.mode()

            if len(mode_diff) == 0:
                return "irregular"

            mode_minutes = mode_diff.iloc[0].total_seconds() / 60

            if mode_minutes < 1:
                return "sub_minute"
            elif mode_minutes == 1:
                return "1min"
            elif mode_minutes == 5:
                return "5min"
            elif mode_minutes == 15:
                return "15min"
            elif mode_minutes == 30:
                return "30min"
            elif mode_minutes == 60:
                return "1hour"
            else:
                return f"{mode_minutes:.1f}min"
        except Exception:
            return "unknown"

    def _resample_by_type(
        self, df_indexed: pd.DataFrame, data_type: str, freq_str: str
    ) -> pd.DataFrame:
        resampled_dict: Dict[str, pd.Series] = {}

        for column in df_indexed.columns:
            if column.endswith("_outlier"):
                resampled_dict[column] = df_indexed[column].resample(freq_str).max()
            elif column == "heart_rate":
                resampled_dict[column] = df_indexed[column].resample(freq_str).mean()
            elif column == "step_count":
                resampled_dict[column] = df_indexed[column].resample(freq_str).sum()
            elif column == "duration_minutes":
                resampled_dict[column] = df_indexed[column].resample(freq_str).sum()
            elif column == "sleep_stage":
                resampled_dict[column] = df_indexed[column].resample(freq_str).agg(
                    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
                )
            else:
                if df_indexed[column].dtype in ["int64", "float64"]:
                    resampled_dict[column] = df_indexed[column].resample(freq_str).mean()
                else:
                    resampled_dict[column] = df_indexed[column].resample(freq_str).first()

        return pd.DataFrame(resampled_dict)

    def _fill_missing_after_resample(
        self, df: pd.DataFrame, data_type: str, fill_method: str
    ) -> Tuple[pd.DataFrame, int]:
        initial_missing = df.isnull().sum().sum()

        if fill_method == "interpolate":
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if not col.endswith("_outlier"):
                    df[col] = df[col].interpolate(
                        method="linear", limit_direction="both"
                    )

            categorical_columns = df.select_dtypes(exclude=[np.number]).columns
            for col in categorical_columns:
                df[col] = df[col].fillna(method="ffill").fillna(method="bfill")

        elif fill_method == "forward_fill":
            df = df.fillna(method="ffill")
        elif fill_method == "backward_fill":
            df = df.fillna(method="bfill")
        elif fill_method == "zero":
            df = df.fillna(0)
        elif fill_method == "drop":
            df = df.dropna()

        final_missing = df.isnull().sum().sum()
        gaps_filled = int(initial_missing - final_missing)
        return df, gaps_filled

    def generate_alignment_report(self, report: Dict) -> str:
        lines = [
            "TIME ALIGNMENT REPORT",
            "=====================",
            f"Original frequency : {report['original_frequency']}",
            f"Target frequency   : {report['target_frequency']}",
            f"Original rows      : {report['original_rows']}",
            f"Resampled rows     : {report['resampled_rows']}",
            f"Gaps filled        : {report['gaps_filled']}",
            f"Fill method        : {report['method_used']}",
            "",
            f"Status: {'✅ Success' if report['success'] else '❌ Failed'}",
        ]
        if not report.get("success") and "error" in report:
            lines.append(f"Error: {report['error']}")
        return "\n".join(lines)


# ======================================================
# PREPROCESSOR – INTEGRATES A + B + C + UI
# ======================================================
class FitnessDataPreprocessor:
    """Complete preprocessing pipeline for fitness tracker data - Milestone 1"""

    def __init__(self):
        self.uploader = FitnessDataUploader()
        self.validator = FitnessDataValidator()
        self.aligner = TimeAligner()

        self.processing_log: List[str] = []
        self.processed_data: Dict[str, pd.DataFrame] = {}
        self.reports: Dict[str, Dict] = {}

    # --------- CORE PIPELINE ----------
    def run_complete_pipeline(
        self,
        use_sample_data: bool,
        raw_data: Optional[Dict[str, pd.DataFrame]] = None,
        target_frequency: str = "1min",
        fill_method: str = "interpolate",
    ) -> Dict[str, pd.DataFrame]:

        self.processing_log.clear()
        self.log_step("Starting Milestone 1 preprocessing pipeline...")

        if use_sample_data:
            self.log_step("Using built-in sample data (demo mode).")
            raw = self._create_sample_data()
        else:
            self.log_step("Using uploaded data from uploader.")
            raw = raw_data or {}

        if not raw:
            st.error("No data available. Upload files or enable sample data.")
            return {}

        # --- Validation & cleaning (B) ---
        self.log_step("Validating and cleaning data (Component B)...")
        validated_data: Dict[str, pd.DataFrame] = {}
        for data_type, df in raw.items():
            cleaned_df, validation_report = self.validator.validate_and_clean_data(
                df, data_type
            )
            validated_data[data_type] = cleaned_df
            self.reports[f"{data_type}_validation"] = validation_report

            with st.expander(
                f"📋 {data_type.title()} – Validation Report", expanded=False
            ):
                st.text(self.validator.generate_validation_report(validation_report))

        # --- Alignment & resampling (C) ---
        self.log_step("Aligning timestamps & resampling (Component C)...")
        aligned_data: Dict[str, pd.DataFrame] = {}
        for data_type, df in validated_data.items():
            aligned_df, alignment_report = self.aligner.align_and_resample(
                df, data_type, target_frequency, fill_method
            )
            aligned_data[data_type] = aligned_df
            self.reports[f"{data_type}_alignment"] = alignment_report

            with st.expander(
                f"⏰ {data_type.title()} – Time Alignment Report", expanded=False
            ):
                st.text(self.aligner.generate_alignment_report(alignment_report))

        self.log_step("Milestone 1 pipeline completed.")
        self.processed_data = aligned_data
        return aligned_data

    def log_step(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.processing_log.append(f"[{timestamp}] {message}")

    # --------- SAMPLE DATA ----------
    def _create_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Create sample heart rate + steps data"""

        timestamps = pd.date_range(
            start="2024-01-15 08:00:00", end="2024-01-15 12:00:00", freq="30S"
        )
        base_hr = 70
        hr_data = []

        for i, ts in enumerate(timestamps):
            if np.random.random() < 0.05:
                hr_data.append(None)
            else:
                time_factor = np.sin(2 * np.pi * i / 120)
                noise = np.random.normal(0, 5)
                hr = base_hr + 20 * time_factor + noise
                hr_data.append(max(50, min(130, hr)))

        if len(hr_data) > 200:
            hr_data[80] = 250
            hr_data[190] = -5

        heart_rate_df = pd.DataFrame(
            {"timestamp": timestamps, "heart_rate": hr_data}
        )

        step_timestamps = [
            "2024-01-15 08:00:00",
            "2024-01-15 08:00:30",
            "2024-01-15 08:02:15",
            "2024-01-15 08:05:00",
            "2024-01-15 08:05:30",
            "2024-01-15 08:10:00",
        ]

        steps_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(step_timestamps),
                "step_count": [100, 150, 200, 250, 275, 400],
            }
        )

        return {"heart_rate": heart_rate_df, "steps": steps_df}

    # --------- DASHBOARD UI ----------
    def show_data_preview(self):
        """Interactive dashboard-style preview for processed data"""
        if not self.processed_data:
            st.warning("Run the pipeline first to see processed data.")
            return

        st.markdown("### 📊 Processed Data – Analytics View")

        data_type = st.selectbox(
            "Select data stream:", list(self.processed_data.keys())
        )
        df = self.processed_data[data_type]

        # KPI CARDS
        total_rows = len(df)
        missing_pct = (
            df.isnull().sum().sum() / (len(df) * len(df.columns))
        ) * 100
        time_span_hours = (
            df["timestamp"].max() - df["timestamp"].min()
        ).total_seconds() / 3600

        numeric_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if not c.endswith("_outlier")
        ]
        outlier_total = 0
        for c in numeric_cols:
            oc = f"{c}_outlier"
            if oc in df.columns:
                outlier_total += int(df[oc].sum())

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Records", f"{total_rows:,}")
        kpi2.metric("Data Quality", f"{100 - missing_pct:.1f}%")
        kpi3.metric("Time Span (hrs)", f"{time_span_hours:.1f}")
        kpi4.metric("Outliers Flagged", outlier_total)

        st.markdown("---")

        # TABS for visuals
        tab_ts, tab_dist, tab_corr = st.tabs(
            ["📈 Time Series", "📊 Distribution", "🧩 Correlation"]
        )

        # ---------- TIME SERIES TAB ----------
        with tab_ts:
            st.markdown("#### Time Series with Outliers")
            if not numeric_cols:
                st.info("No numeric columns available for plotting.")
            else:
                metric = st.selectbox(
                    "Metric to plot:", numeric_cols, key=f"ts_{data_type}"
                )
                fig = go.Figure()

                # Rolling average for smoother line
                window = st.slider(
                    "Rolling window (minutes)",
                    min_value=1,
                    max_value=30,
                    value=5,
                    help="Applied on resampled data to smooth noise.",
                )
                df_sorted = df.sort_values("timestamp").copy()
                df_sorted[f"{metric}_smooth"] = (
                    df_sorted[metric].rolling(window=window, min_periods=1).mean()
                )

                fig.add_trace(
                    go.Scatter(
                        x=df_sorted["timestamp"],
                        y=df_sorted[metric],
                        mode="lines",
                        name="Raw",
                        opacity=0.3,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted["timestamp"],
                        y=df_sorted[f"{metric}_smooth"],
                        mode="lines",
                        name="Smoothed",
                    )
                )

                outlier_col = f"{metric}_outlier"
                if outlier_col in df_sorted.columns:
                    outlier_data = df_sorted[df_sorted[outlier_col] == True]
                    if not outlier_data.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=outlier_data["timestamp"],
                                y=outlier_data[metric],
                                mode="markers",
                                name="Outliers",
                                marker=dict(color="red", size=9, symbol="x"),
                            )
                        )

                fig.update_layout(
                    height=420,
                    hovermode="x unified",
                    xaxis_title="Timestamp",
                    yaxis_title=metric.replace("_", " ").title(),
                )
                st.plotly_chart(fig, use_container_width=True)

        # ---------- DISTRIBUTION TAB ----------
        with tab_dist:
            st.markdown("#### Value Distribution & Variability")
            if not numeric_cols:
                st.info("No numeric columns available.")
            else:
                metric = st.selectbox(
                    "Metric:", numeric_cols, key=f"dist_{data_type}"
                )

                col_hist, col_box = st.columns([2, 1])

                with col_hist:
                    fig_hist = px.histogram(
                        df,
                        x=metric,
                        nbins=40,
                        marginal="rug",
                        title=f"{metric.replace('_', ' ').title()} Distribution",
                    )
                    fig_hist.update_layout(height=350)
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col_box:
                    fig_box = px.box(
                        df,
                        y=metric,
                        points="all",
                        title="Boxplot (spread & outliers)",
                    )
                    fig_box.update_layout(height=350)
                    st.plotly_chart(fig_box, use_container_width=True)

                st.markdown("##### Quick Stats")
                stats = df[metric].describe().to_frame().T
                st.dataframe(stats, use_container_width=True)

        # ---------- CORRELATION TAB ----------
        with tab_corr:
            st.markdown("#### Correlation Between Metrics")
            if len(numeric_cols) < 2:
                st.info("Need at least two numeric columns to compute correlation.")
            else:
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Heatmap",
                )
                fig_corr.update_layout(height=420)
                st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("#### Sample of Processed Data")
        st.dataframe(df.head(200), use_container_width=True)

    def show_logs(self):
        st.markdown("### 📜 Processing Log")
        if not self.processing_log:
            st.info("Run the pipeline to see logs.")
        else:
            for line in self.processing_log:
                st.text(line)


# ======================================================
# STREAMLIT PAGE ENTRYPOINT
# ======================================================
def main():
    st.set_page_config(
        page_title="FitPulse – Milestone 1",
        page_icon="📊",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🏃‍♂️ FitPulse – Anomaly Detection")
    st.caption(
        "Upload raw fitness tracker data, clean it, align timestamps, and explore "
        "professional visualizations – ready for feature extraction and anomaly detection."
    )

    if "preprocessor" not in st.session_state:
        st.session_state.preprocessor = FitnessDataPreprocessor()
    preprocessor: FitnessDataPreprocessor = st.session_state.preprocessor

    # Initialize session state for data persistence
    if "raw_fitness_data" not in st.session_state:
        st.session_state.raw_fitness_data = None
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = {}

    # ---------- SIDEBAR ----------
    st.sidebar.header("⚙️ Pipeline Settings")

    use_sample_data = st.sidebar.checkbox(
        "Use sample demo data", value=False, help="Turn off to upload your own files."
    )

    target_frequency = st.sidebar.selectbox(
        "Resampling frequency", ["1min", "5min", "15min", "30min", "1hour"], index=0
    )

    fill_method = st.sidebar.selectbox(
        "Missing values handling",
        ["interpolate", "forward_fill", "backward_fill", "zero", "drop"],
        index=0,
    )

   
    # ---------- MAIN TABS ----------
    tab_run, tab_preview, tab_logs = st.tabs(
        ["🚀 Run Pipeline", "📊 Visual Analytics", "📜 Logs"]
    )

    with tab_run:
        st.markdown("### 1️⃣ Run Preprocessing Pipeline")

        raw_data = None
        if use_sample_data:
            st.info(
                "Sample heart-rate and steps data will be generated automatically.\n\n"
                "Use this mode to demo the full pipeline without real device exports."
            )
        else:
            raw_data = preprocessor.uploader.create_upload_interface()

        if st.button("🚀 Run Pipeline", type="primary"):
            with st.spinner("Processing data through A+B+C pipeline..."):
                processed = preprocessor.run_complete_pipeline(
                    use_sample_data=use_sample_data,
                    raw_data=raw_data,
                    target_frequency=target_frequency,
                    fill_method=fill_method,
                )

            if processed:
                # Store processed data in session state for persistence across pages
                st.session_state.processed_data = processed
                st.session_state.raw_fitness_data = raw_data
                st.success("✅ Milestone 1 completed successfully.")
                st.info("Data stored! You can now switch to other pages.")
            else:
                st.error("❌ Pipeline did not complete. Check messages above.")

    with tab_preview:
        preprocessor.show_data_preview()

    with tab_logs:
        preprocessor.show_logs()


if __name__ == "__main__":
    main()
