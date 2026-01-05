import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import backend pipeline
from milestone_Pipeline import AnomalyDetectionPipeline


def main():
    st.set_page_config(
        page_title="Anomaly Detection",
        page_icon="🚨",
        layout="wide"
    )

    st.title("🚨 Anomaly Detection & Visualization")
    st.caption("Detect unusual heart-rate, steps & sleep patterns using multiple AI methods")

    # Initialize pipeline object
    if "milestone3_pipeline" not in st.session_state:
        st.session_state.milestone3_pipeline = AnomalyDetectionPipeline()

    pipeline: AnomalyDetectionPipeline = st.session_state.milestone3_pipeline

    # ---------------- LOAD PREVIOUS DATA ----------------
    preprocessed_data = None
    prophet_forecasts = None
    cluster_labels = None
    feature_matrices = None

    if "preprocessor" in st.session_state:
        preprocessed_data = st.session_state.preprocessor.processed_data
        st.success("📌 Loaded processed data")

    if "milestone2_results" in st.session_state:
        m2 = st.session_state.milestone2_results
        prophet_forecasts = m2.get("forecasts")
        feature_matrices = m2.get("features")
        cluster_labels = m2.get("clusters")
        st.success("📌 Loaded Prophet & Cluster outputs")

    if not preprocessed_data:
        st.warning("⚠ No data found – Run Step-1 & 2 first")
        return

    # ---------------- SETTINGS ----------------
    st.sidebar.header("⚙ Settings")
    std = st.sidebar.slider("Residual Threshold (std)", 1.0, 6.0, 3.0, 0.5)

    # ---------------- RUN ANOMALY DETECTION ----------------
    if st.button("🚀 Run Anomaly Detection", type="primary"):
        with st.spinner("Detecting anomalies..."):
            pipeline.residual.threshold_std = std
            results = pipeline.run_complete_milestone3(
                preprocessed_data=preprocessed_data,
                prophet_forecasts=prophet_forecasts,
                cluster_labels=cluster_labels,
                feature_matrices=feature_matrices
            )
            st.session_state.milestone3_results = results

        st.success("🎉 Anomaly Detection Completed!")

    # ---------------- DISPLAY RESULTS ----------------
    if "milestone3_results" in st.session_state:
        res = st.session_state.milestone3_results
        reports = res.get("reports", {})

        # ---------------- KPI COUNT CARDS ----------------
        st.markdown("### 📊 Detection Summary")

        heart = reports.get("heart_rate", {}).get("threshold", {}).get("anomalies_detected", 0)
        steps = reports.get("steps", {}).get("threshold", {}).get("anomalies_detected", 0)
        sleep = reports.get("sleep", {}).get("threshold", {}).get("anomalies_detected", 0)

        c1, c2, c3 = st.columns(3)
        c1.metric("❤️ Heart-rate Anomalies", heart)
        c2.metric("👟 Step-count Anomalies", steps)
        c3.metric("😴 Sleep Anomalies", sleep)
        st.markdown("---")

        # ---------------- TIME-SERIES VISUALIZATIONS ----------------
        st.markdown("### 📈 Time-series Visualizations")

        def _plot_anomalies_ts(df, metric_col, title):
            if df is None or metric_col not in df.columns or 'timestamp' not in df.columns:
                return
            d = df.copy()
            d['timestamp'] = pd.to_datetime(d['timestamp'])
            d = d.sort_values('timestamp')

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d['timestamp'], y=d[metric_col], mode='lines', name='actual', line=dict(color='blue')))

            if 'predicted' in d.columns:
                fig.add_trace(go.Scatter(x=d['timestamp'], y=d['predicted'], mode='lines', name='predicted', line=dict(color='orange', dash='dash')))

            # Add anomaly markers for each method if present
            if 'threshold_anomaly' in d.columns:
                thr = d[d['threshold_anomaly'] == True]
                fig.add_trace(go.Scatter(x=thr['timestamp'], y=thr[metric_col], mode='markers', name='threshold', marker=dict(color='red', size=8, symbol='circle-open')))
            if 'residual_anomaly' in d.columns:
                res = d[d['residual_anomaly'] == True]
                fig.add_trace(go.Scatter(x=res['timestamp'], y=res[metric_col], mode='markers', name='residual', marker=dict(color='purple', size=8, symbol='x')))
            if 'cluster_anomaly' in d.columns:
                clu = d[d['cluster_anomaly'] == True]
                fig.add_trace(go.Scatter(x=clu['timestamp'], y=clu[metric_col], mode='markers', name='cluster', marker=dict(color='green', size=8, symbol='diamond')))

            fig.update_layout(title=title, xaxis_title='Timestamp', yaxis_title=metric_col.replace('_', ' ').title(), height=350, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
            st.plotly_chart(fig, use_container_width=True)

        # Plot each available metric
        metric_map = {'heart_rate': 'heart_rate', 'steps': 'step_count', 'sleep': 'duration_minutes'}
        for key, metric in metric_map.items():
            df_full = res.get('data', {}).get(key)
            if df_full is not None:
                title = f"{key.replace('_', ' ').title()} — Actual vs Predicted & Anomalies"
                _plot_anomalies_ts(df_full, metric, title)


        # ---------------- DETAILED ANOMALY TABLES ----------------
        st.markdown("### 📋 Detailed Anomaly Analysis")
        
        anomaly_data = res.get("data", {})
        
        # Heart Rate Anomalies
        if "heart_rate" in anomaly_data:
            with st.expander("❤️ Heart Rate Anomalies", expanded=False):
                hr_df = anomaly_data["heart_rate"].copy()
                
                # Filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_threshold = st.checkbox("Show Threshold Anomalies", value=True, key="hr_threshold")
                with col2:
                    show_residual = st.checkbox("Show Residual Anomalies", value=True, key="hr_residual")
                with col3:
                    show_cluster = st.checkbox("Show Cluster Anomalies", value=True, key="hr_cluster")
                
                # Filter dataframe based on selections
                filtered_hr = hr_df.copy()
                anomaly_mask = pd.Series([False] * len(filtered_hr), index=filtered_hr.index)
                
                if show_threshold and 'threshold_anomaly' in filtered_hr.columns:
                    anomaly_mask |= filtered_hr['threshold_anomaly']
                if show_residual and 'residual_anomaly' in filtered_hr.columns:
                    anomaly_mask |= filtered_hr['residual_anomaly']
                if show_cluster and 'cluster_anomaly' in filtered_hr.columns:
                    anomaly_mask |= filtered_hr['cluster_anomaly']
                
                filtered_hr = filtered_hr[anomaly_mask]
                
                if len(filtered_hr) > 0:
                    # Select relevant columns for display
                    display_cols = ['timestamp', 'heart_rate', 'threshold_anomaly', 'residual_anomaly', 'cluster_anomaly']
                    display_cols = [col for col in display_cols if col in filtered_hr.columns]
                    
                    st.dataframe(
                        filtered_hr[display_cols].style.highlight_max(axis=0, subset=['heart_rate']),
                        use_container_width=True,
                        height=300
                    )
                    st.info(f"🔍 Found {len(filtered_hr)} anomalies")
                else:
                    st.success("✅ No anomalies detected for the selected filters")
        
        # Steps Anomalies
        if "steps" in anomaly_data:
            with st.expander("👟 Step Count Anomalies", expanded=False):
                steps_df = anomaly_data["steps"].copy()
                
                # Filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_threshold = st.checkbox("Show Threshold Anomalies", value=True, key="steps_threshold")
                with col2:
                    show_residual = st.checkbox("Show Residual Anomalies", value=True, key="steps_residual")
                with col3:
                    show_cluster = st.checkbox("Show Cluster Anomalies", value=True, key="steps_cluster")
                
                # Filter dataframe based on selections
                filtered_steps = steps_df.copy()
                anomaly_mask = pd.Series([False] * len(filtered_steps), index=filtered_steps.index)
                
                if show_threshold and 'threshold_anomaly' in filtered_steps.columns:
                    anomaly_mask |= filtered_steps['threshold_anomaly']
                if show_residual and 'residual_anomaly' in filtered_steps.columns:
                    anomaly_mask |= filtered_steps['residual_anomaly']
                if show_cluster and 'cluster_anomaly' in filtered_steps.columns:
                    anomaly_mask |= filtered_steps['cluster_anomaly']
                
                filtered_steps = filtered_steps[anomaly_mask]
                
                if len(filtered_steps) > 0:
                    # Select relevant columns for display
                    display_cols = ['timestamp', 'step_count', 'threshold_anomaly', 'residual_anomaly', 'cluster_anomaly']
                    display_cols = [col for col in display_cols if col in filtered_steps.columns]
                    
                    st.dataframe(
                        filtered_steps[display_cols].style.highlight_max(axis=0, subset=['step_count']),
                        use_container_width=True,
                        height=300
                    )
                    st.info(f"🔍 Found {len(filtered_steps)} anomalies")
                else:
                    st.success("✅ No anomalies detected for the selected filters")
        
        # Sleep Anomalies
        if "sleep" in anomaly_data:
            with st.expander("😴 Sleep Anomalies", expanded=False):
                sleep_df = anomaly_data["sleep"].copy()
                
                # Filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_threshold = st.checkbox("Show Threshold Anomalies", value=True, key="sleep_threshold")
                with col2:
                    show_residual = st.checkbox("Show Residual Anomalies", value=True, key="sleep_residual")
                with col3:
                    show_cluster = st.checkbox("Show Cluster Anomalies", value=True, key="sleep_cluster")
                
                # Filter dataframe based on selections
                filtered_sleep = sleep_df.copy()
                anomaly_mask = pd.Series([False] * len(filtered_sleep), index=filtered_sleep.index)
                
                if show_threshold and 'threshold_anomaly' in filtered_sleep.columns:
                    anomaly_mask |= filtered_sleep['threshold_anomaly']
                if show_residual and 'residual_anomaly' in filtered_sleep.columns:
                    anomaly_mask |= filtered_sleep['residual_anomaly']
                if show_cluster and 'cluster_anomaly' in filtered_sleep.columns:
                    anomaly_mask |= filtered_sleep['cluster_anomaly']
                
                filtered_sleep = filtered_sleep[anomaly_mask]
                
                if len(filtered_sleep) > 0:
                    # Select relevant columns for display
                    display_cols = ['timestamp', 'duration_minutes', 'threshold_anomaly', 'residual_anomaly', 'cluster_anomaly']
                    display_cols = [col for col in display_cols if col in filtered_sleep.columns]
                    
                    st.dataframe(
                        filtered_sleep[display_cols].style.highlight_max(axis=0, subset=['duration_minutes']),
                        use_container_width=True,
                        height=300
                    )
                    st.info(f"🔍 Found {len(filtered_sleep)} anomalies")
                else:
                    st.success("✅ No anomalies detected for the selected filters")
        
        st.markdown("---")
        
     


if __name__ == "__main__":
    main()
