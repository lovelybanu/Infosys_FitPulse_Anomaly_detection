import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ===============================
# Dashboard – Final Business Layer
# ===============================

def main():
    st.set_page_config(page_title="📊 FitPulse Dashboard", page_icon="📊", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .kpi-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-label { font-size: 14px; opacity: 0.8; }
        .metric-value { font-size: 32px; font-weight: bold; }
        .anomaly-badge { display: inline-block; padding: 5px 10px; border-radius: 5px; font-size: 12px; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 FitPulse – Advanced Analytics Dashboard")
    st.caption("🔍 Comprehensive anomaly detection & health insights powered by AI")

    # load anomaly results from milestone-3
    if "milestone3_results" not in st.session_state:
        st.warning("⚠ No anomaly results found. Run Milestone-3 first.")
        return

    results = st.session_state.milestone3_results
    data_dict = results.get("data", {})
    reports = results.get("reports", {})

    # Calculate comprehensive stats
    def calc_stats(data_type):
        df = data_dict.get(data_type)
        if df is None or df.empty:
            return {"total": 0, "threshold": 0, "residual": 0, "cluster": 0, "percentage": 0, "any_anomaly": 0}
        
        total = len(df)
        threshold_count = 0
        residual_count = 0
        cluster_count = 0
        
        # Safely check for column existence and count anomalies
        if "threshold_anomaly" in df.columns:
            threshold_count = (df["threshold_anomaly"] == True).sum()
        if "residual_anomaly" in df.columns:
            residual_count = (df["residual_anomaly"] == True).sum()
        if "cluster_anomaly" in df.columns:
            cluster_count = (df["cluster_anomaly"] == True).sum()
        
        # Count any anomaly across all methods
        any_anomaly = 0
        if "threshold_anomaly" in df.columns or "residual_anomaly" in df.columns or "cluster_anomaly" in df.columns:
            anomaly_mask = pd.Series([False] * len(df), index=df.index)
            if "threshold_anomaly" in df.columns:
                anomaly_mask |= (df["threshold_anomaly"] == True)
            if "residual_anomaly" in df.columns:
                anomaly_mask |= (df["residual_anomaly"] == True)
            if "cluster_anomaly" in df.columns:
                anomaly_mask |= (df["cluster_anomaly"] == True)
            any_anomaly = anomaly_mask.sum()
        
        return {
            "total": total,
            "threshold": int(threshold_count),
            "residual": int(residual_count),
            "cluster": int(cluster_count),
            "any_anomaly": int(any_anomaly),
            "percentage": round((any_anomaly / total * 100) if total > 0 else 0, 2)
        }
    
    hr_stats = calc_stats("heart_rate")
    steps_stats = calc_stats("steps")
    sleep_stats = calc_stats("sleep")

    # ======================
    # ENHANCED KPI SUMMARY
    # ======================
    st.markdown("## 📌 Key Performance Indicators")

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3, gap="medium")
    
    with kpi_col1:
        st.metric(
            "❤️ Heart Rate",
            f"{hr_stats['any_anomaly']}",
            f"{hr_stats['percentage']}% anomalous"
        )
        with st.expander("Details", expanded=False):
            st.caption(f"Total records: {hr_stats['total']}")
            st.caption(f"🔴 Threshold: {hr_stats['threshold']}")
            st.caption(f"🟠 Residual: {hr_stats['residual']}")
            st.caption(f"🟢 Cluster: {hr_stats['cluster']}")
    
    with kpi_col2:
        st.metric(
            "👟 Steps",
            f"{steps_stats['any_anomaly']}",
            f"{steps_stats['percentage']}% anomalous"
        )
        with st.expander("Details", expanded=False):
            st.caption(f"Total records: {steps_stats['total']}")
            st.caption(f"🔴 Threshold: {steps_stats['threshold']}")
            st.caption(f"🟠 Residual: {steps_stats['residual']}")
            st.caption(f"🟢 Cluster: {steps_stats['cluster']}")
    
    with kpi_col3:
        st.metric(
            "😴 Sleep",
            f"{sleep_stats['any_anomaly']}",
            f"{sleep_stats['percentage']}% anomalous"
        )
        with st.expander("Details", expanded=False):
            st.caption(f"Total records: {sleep_stats['total']}")
            st.caption(f"🔴 Threshold: {sleep_stats['threshold']}")
            st.caption(f"🟠 Residual: {sleep_stats['residual']}")
            st.caption(f"🟢 Cluster: {sleep_stats['cluster']}")

    st.markdown("---")

    # ======================
    # MAIN VISUAL PANEL WITH TABS
    # ======================
    st.markdown("## 📈 Advanced Visual Analytics")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Trends & Anomalies", "🔍 Anomaly Breakdown", "📉 Distribution", "💡 Insights"])

    with tab1:
        st.markdown("### Time-Series Analysis with Anomaly Markers")
        
        metric_options = {"Heart Rate": "heart_rate", "Steps": "steps", "Sleep Duration": "sleep"}
        col1, col2 = st.columns([3, 1])
        with col1:
            selected = st.selectbox("Choose metric", metric_options.keys(), key="metric_select")
        with col2:
            show_anom = st.checkbox("Anomalies only", value=False)

        key = metric_options[selected]
        df = data_dict.get(key)

        if df is None or df.empty:
            st.info("No data found for selected metric.")
        else:
            df_plot = df.copy()
            if show_anom:
                mask = (
                    (df_plot.get("threshold_anomaly") == True) |
                    (df_plot.get("residual_anomaly") == True) |
                    (df_plot.get("cluster_anomaly") == True)
                )
                df_plot = df_plot[mask]

            df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
            df_plot = df_plot.sort_values("timestamp")

            if not df_plot.empty:
                metric_col = df_plot.columns[1]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_plot["timestamp"], y=df_plot[metric_col],
                    mode="lines", name="Actual",
                    line=dict(color="rgba(0,100,200,0.8)", width=2)
                ))

                if "predicted" in df_plot.columns:
                    fig.add_trace(go.Scatter(
                        x=df_plot["timestamp"], y=df_plot["predicted"],
                        mode="lines", name="Predicted",
                        line=dict(color="rgba(255,140,0,0.6)", dash="dash", width=2)
                    ))

                # Anomaly markers
                anomaly_colors = {
                    "threshold_anomaly": ("red", "🔴 Threshold"),
                    "residual_anomaly": ("purple", "🟣 Residual"),
                    "cluster_anomaly": ("green", "🟢 Cluster")
                }
                
                for col, (color, label) in anomaly_colors.items():
                    if col in df_plot.columns:
                        pts = df_plot[df_plot[col] == True]
                        if not pts.empty:
                            fig.add_trace(go.Scatter(
                                x=pts["timestamp"], y=pts[metric_col],
                                mode="markers", name=label,
                                marker=dict(color=color, size=10, symbol="circle-open", line=dict(width=2))
                            ))

                fig.update_layout(
                    height=400, 
                    title=f"<b>{selected} – Trend & Anomaly Detection</b>",
                    xaxis_title="Timestamp",
                    yaxis_title=selected,
                    hovermode="x unified",
                    plot_bgcolor="rgba(240,240,250,0.5)",
                    font=dict(size=11)
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Anomaly Detection Method Breakdown")
        
        # Create anomaly breakdown chart
        anomaly_breakdown = {
            "Threshold": [hr_stats['threshold'], steps_stats['threshold'], sleep_stats['threshold']],
            "Residual": [hr_stats['residual'], steps_stats['residual'], sleep_stats['residual']],
            "Cluster": [hr_stats['cluster'], steps_stats['cluster'], sleep_stats['cluster']]
        }
        
        breakdown_df = pd.DataFrame(anomaly_breakdown, index=["Heart Rate", "Steps", "Sleep"])
        
        fig_breakdown = go.Figure(data=[
            go.Bar(name="Threshold", x=breakdown_df.index, y=breakdown_df["Threshold"], marker_color="rgba(255,0,0,0.7)"),
            go.Bar(name="Residual", x=breakdown_df.index, y=breakdown_df["Residual"], marker_color="rgba(255,165,0,0.7)"),
            go.Bar(name="Cluster", x=breakdown_df.index, y=breakdown_df["Cluster"], marker_color="rgba(0,128,0,0.7)")
        ])
        fig_breakdown.update_layout(barmode='group', height=350, title="<b>Anomaly Detection Method Comparison</b>", 
                                   yaxis_title="Count", hovermode="x unified")
        st.plotly_chart(fig_breakdown, use_container_width=True)
        
        # Anomaly percentage pie charts
        col_pie1, col_pie2, col_pie3 = st.columns(3)
        
        with col_pie1:
            if hr_stats['any_anomaly'] > 0:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=["Threshold", "Residual", "Cluster"],
                    values=[hr_stats['threshold'], hr_stats['residual'], hr_stats['cluster']],
                    marker=dict(colors=['red', 'orange', 'green'])
                )])
                fig_pie.update_layout(height=300, title="<b>Heart Rate</b>")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("✅ No anomalies detected")
        
        with col_pie2:
            if steps_stats['any_anomaly'] > 0:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=["Threshold", "Residual", "Cluster"],
                    values=[steps_stats['threshold'], steps_stats['residual'], steps_stats['cluster']],
                    marker=dict(colors=['red', 'orange', 'green'])
                )])
                fig_pie.update_layout(height=300, title="<b>Steps</b>")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("✅ No anomalies detected")
        
        with col_pie3:
            if sleep_stats['any_anomaly'] > 0:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=["Threshold", "Residual", "Cluster"],
                    values=[sleep_stats['threshold'], sleep_stats['residual'], sleep_stats['cluster']],
                    marker=dict(colors=['red', 'orange', 'green'])
                )])
                fig_pie.update_layout(height=300, title="<b>Sleep</b>")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("✅ No anomalies detected")

    with tab3:
        st.markdown("### Value Distribution Analysis")
        
        dist_metric = st.selectbox("Select metric for distribution", 
                                   ["Heart Rate", "Steps", "Sleep Duration"],
                                   key="dist_select")
        
        dist_key_map = {"Heart Rate": "heart_rate", "Steps": "steps", "Sleep Duration": "sleep"}
        dist_df = data_dict.get(dist_key_map[dist_metric])
        
        if dist_df is not None and not dist_df.empty:
            metric_col = dist_df.columns[1]
            
            # Histogram
            fig_hist = px.histogram(dist_df, x=metric_col, nbins=30, 
                                   title=f"<b>{dist_metric} – Distribution</b>",
                                   color_discrete_sequence=['steelblue'])
            fig_hist.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Statistics
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Mean", f"{dist_df[metric_col].mean():.2f}")
            col_s2.metric("Median", f"{dist_df[metric_col].median():.2f}")
            col_s3.metric("Std Dev", f"{dist_df[metric_col].std():.2f}")
            col_s4.metric("Range", f"{dist_df[metric_col].max() - dist_df[metric_col].min():.2f}")

    with tab4:
        st.markdown("### 🧠 AI-Powered Health Insights & Recommendations")
        
        # Calculate health score
        total_metrics = hr_stats['total'] + steps_stats['total'] + sleep_stats['total']
        total_anomalies = hr_stats['any_anomaly'] + steps_stats['any_anomaly'] + sleep_stats['any_anomaly']
        health_score = max(0, 100 - (total_anomalies / total_metrics * 100)) if total_metrics > 0 else 100
        
        score_col, gauge_col = st.columns([1, 2])
        with score_col:
            st.metric("Health Score", f"{health_score:.1f}%", 
                     delta="Good" if health_score >= 70 else "Fair" if health_score >= 50 else "Poor",
                     delta_color="normal" if health_score >= 70 else "off")
        
        with gauge_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=health_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Wellness"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("---")
        
        # Smart recommendations
        insights = []
        
        # Heart Rate Insights
        if hr_stats['percentage'] > 15:
            insights.append(("🚨 CRITICAL", "❤️ High heart-rate anomaly rate (>15%). Consider medical check-up.", "red"))
        elif hr_stats['percentage'] > 5:
            insights.append(("⚠️ WARNING", "❤️ Elevated heart-rate anomalies. Check stress levels & sleep quality.", "orange"))
        elif hr_stats['any_anomaly'] == 0:
            insights.append(("✅ GOOD", "❤️ Heart rate stable and normal. Keep up the healthy lifestyle.", "green"))
        
        # Steps Insights
        if steps_stats['percentage'] == 0:
            insights.append(("✅ GOOD", "👟 Step count is consistently normal. Activity level is healthy.", "green"))
        elif steps_stats['percentage'] > 10:
            insights.append(("⚠️ WARNING", "👟 Unusual step patterns detected. Review your daily activity logs.", "orange"))
        
        # Sleep Insights
        if sleep_stats['percentage'] > 20:
            insights.append(("🚨 CRITICAL", "😴 Significant sleep irregularities (>20%). Improve sleep hygiene.", "red"))
        elif sleep_stats['percentage'] > 5:
            insights.append(("⚠️ WARNING", "😴 Sleep duration varies. Try maintaining a consistent sleep schedule.", "orange"))
        elif sleep_stats['any_anomaly'] == 0:
            insights.append(("✅ GOOD", "😴 Sleep patterns are healthy and consistent. Great work!", "green"))
        
        if not insights:
            st.success("🎉 Everything looks normal! Continue maintaining your healthy habits.")
        else:
            for level, message, color in insights:
                with st.container():
                    st.markdown(f"<div style='background-color: {color}; color: white; padding: 15px; border-radius: 5px; margin: 10px 0;'>"
                               f"<b>{level}</b> {message}</div>", unsafe_allow_html=True)
        
        # Trend indicators
        st.markdown("### 📊 Trend Summary")
        trend_col1, trend_col2, trend_col3 = st.columns(3)
        
        with trend_col1:
            trend = "📈 Worsening" if hr_stats['percentage'] > 10 else "📉 Improving" if hr_stats['percentage'] < 3 else "➡️ Stable"
            st.info(f"**Heart Rate Trend:** {trend}\nAnomalies: {hr_stats['percentage']}%")
        
        with trend_col2:
            trend = "📈 Worsening" if steps_stats['percentage'] > 5 else "📉 Improving" if steps_stats['percentage'] == 0 else "➡️ Stable"
            st.info(f"**Steps Trend:** {trend}\nAnomalies: {steps_stats['percentage']}%")
        
        with trend_col3:
            trend = "📈 Worsening" if sleep_stats['percentage'] > 10 else "📉 Improving" if sleep_stats['percentage'] < 5 else "➡️ Stable"
            st.info(f"**Sleep Trend:** {trend}\nAnomalies: {sleep_stats['percentage']}%")

    st.markdown("---")

    # ======================
    # DOWNLOAD SECTION
    # ======================
    st.markdown("## 📥 Export & Data Management")
    
    export_col1, export_col2 = st.columns([2, 1])
    
    with export_col1:
        st.markdown("### 📋 Complete Anomaly Dataset")
        
        export_df_list = []
        for k, dfv in data_dict.items():
            dfv = dfv.copy()
            dfv["type"] = k
            export_df_list.append(dfv)

        export_big = pd.concat(export_df_list)
        export_big = export_big[
            ["timestamp", "type"] + [c for c in export_big.columns if c not in ["timestamp","type"]]
        ]

        st.dataframe(export_big.head(50), use_container_width=True, height=300)
    
    with export_col2:
        st.markdown("### 💾 Download Options")
        st.download_button(
            "📥 CSV Report",
            export_big.to_csv(index=False).encode(),
            "fitpulse_anomaly_report.csv",
            "text/csv",
            use_container_width=True
        )
        st.download_button(
            "📊 Excel Report",
            export_big.to_excel(index=False, engine='openpyxl') if 'openpyxl' in __import__('sys').modules else export_big.to_csv(index=False).encode(),
            "fitpulse_anomaly_report.xlsx" if 'openpyxl' in __import__('sys').modules else "fitpulse_anomaly_report.csv",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if 'openpyxl' in __import__('sys').modules else "text/csv",
            use_container_width=True
        )

    st.markdown("---")
    
    # ======================
    # FOOTER
    # ======================
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
    <small>
    🔬 <b>FitPulse Anomaly Detection</b> | Powered by Threshold, Prophet Residual & Clustering Analysis<br>
    </b>
    ✨ Stay healthy, stay monitored!
    </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
