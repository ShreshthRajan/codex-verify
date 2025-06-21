# ui/components/metrics_charts.py
"""
Metrics Charts Component - Interactive charts and visualizations.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any
import numpy as np

class MetricsChartsComponent:
   """Component for rendering interactive charts and metrics"""
   
   def render_result_charts(self, result):
       """Render charts for single verification result"""
       
       # Agent performance radar chart
       self._render_agent_radar_chart(result)
       
       # Issue distribution charts
       col1, col2 = st.columns(2)
       
       with col1:
           self._render_severity_distribution(result)
       
       with col2:
           self._render_issue_types_chart(result)
       
       # Performance metrics
       self._render_performance_metrics(result)
   
   def _render_agent_radar_chart(self, result):
       """Render radar chart of agent performance"""
       
       agents = []
       scores = []
       
       for agent_name, agent_result in result.agent_results.items():
           agents.append(agent_name.title())
           scores.append(agent_result.overall_score)
       
       # Add overall score
       agents.append("Overall")
       scores.append(result.overall_score)
       
       fig = go.Figure()
       
       fig.add_trace(go.Scatterpolar(
           r=scores,
           theta=agents,
           fill='toself',
           name='Verification Scores',
           line_color='#2E86AB'
       ))
       
       fig.update_layout(
           polar=dict(
               radialaxis=dict(
                   visible=True,
                   range=[0, 1]
               )),
           showlegend=False,
           title="Agent Performance Overview",
           height=400
       )
       
       st.plotly_chart(fig, use_container_width=True)
   
   def _render_severity_distribution(self, result):
       """Render pie chart of issue severity distribution"""
       
       severity_counts = {}
       for issue in result.aggregated_issues:
           severity = issue.severity.value.title()
           severity_counts[severity] = severity_counts.get(severity, 0) + 1
       
       if severity_counts:
           fig = px.pie(
               values=list(severity_counts.values()),
               names=list(severity_counts.keys()),
               title="Issues by Severity",
               color_discrete_map={
                   'Critical': '#dc3545',
                   'High': '#fd7e14', 
                   'Medium': '#ffc107',
                   'Low': '#28a745'
               }
           )
           
           st.plotly_chart(fig, use_container_width=True)
       else:
           st.success("🎉 No issues detected!")
   
   def _render_issue_types_chart(self, result):
       """Render bar chart of issue types"""
       
       type_counts = {}
       for issue in result.aggregated_issues:
           issue_type = issue.type.replace('_', ' ').title()
           type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
       
       if type_counts:
           # Sort by count
           sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
           
           fig = px.bar(
               x=[count for _, count in sorted_types],
               y=[type_name for type_name, _ in sorted_types],
               orientation='h',
               title="Top Issue Types",
               labels={'x': 'Count', 'y': 'Issue Type'}
           )
           
           fig.update_layout(height=400)
           st.plotly_chart(fig, use_container_width=True)
       else:
           st.info("No issue type data available")
   
   def _render_performance_metrics(self, result):
       """Render performance timing metrics"""
       
       # Create performance data
       perf_data = []
       for agent_name, agent_result in result.agent_results.items():
           perf_data.append({
               'Agent': agent_name.title(),
               'Execution Time (ms)': agent_result.execution_time * 1000,
               'Issues Found': len(agent_result.issues),
               'Score': agent_result.overall_score
           })
       
       df = pd.DataFrame(perf_data)
       
       # Create subplot with secondary y-axis
       fig = make_subplots(
           rows=1, cols=2,
           subplot_titles=('Execution Time', 'Performance vs Issues'),
           specs=[[{"secondary_y": False}, {"secondary_y": True}]]
       )
       
       # Execution time bar chart
       fig.add_trace(
           go.Bar(x=df['Agent'], y=df['Execution Time (ms)'], name='Execution Time'),
           row=1, col=1
       )
       
       # Performance vs issues scatter plot
       fig.add_trace(
           go.Scatter(
               x=df['Issues Found'], 
               y=df['Score'],
               mode='markers+text',
               text=df['Agent'],
               textposition='top center',
               marker=dict(size=10, color=df['Score'], colorscale='RdYlGn'),
               name='Agent Performance'
           ),
           row=1, col=2
       )
       
       fig.update_layout(
           height=400,
           title_text="Performance Analysis",
           showlegend=False
       )
       
       fig.update_xaxes(title_text="Agent", row=1, col=1)
       fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
       fig.update_xaxes(title_text="Issues Found", row=1, col=2)
       fig.update_yaxes(title_text="Score", row=1, col=2)
       
       st.plotly_chart(fig, use_container_width=True)
   
   def render_analytics_dashboard(self, verification_history):
       """Render analytics dashboard from verification history"""
       
       if len(verification_history) < 2:
           st.info("Need at least 2 verifications to show trends")
           return
       
       # Prepare trend data
       trend_data = []
       for i, entry in enumerate(verification_history):
           result = entry['result']
           trend_data.append({
               'Verification': i + 1,
               'Overall Score': result.overall_score,
               'Critical Issues': len([iss for iss in result.aggregated_issues if iss.severity.value == 'critical']),
               'High Issues': len([iss for iss in result.aggregated_issues if iss.severity.value == 'high']),
               'Total Issues': len(result.aggregated_issues),
               'Execution Time': result.execution_time
           })
       
       df = pd.DataFrame(trend_data)
       
       # Score trends
       col1, col2 = st.columns(2)
       
       with col1:
           fig = px.line(df, x='Verification', y='Overall Score',
                        title="Score Trend Over Time",
                        markers=True)
           fig.add_hline(y=0.85, line_dash="dash", line_color="green", 
                        annotation_text="Enterprise Threshold")
           st.plotly_chart(fig, use_container_width=True)
       
       with col2:
           fig = px.bar(df, x='Verification', y=['Critical Issues', 'High Issues'],
                       title="Issue Trends",
                       color_discrete_map={'Critical Issues': '#dc3545', 'High Issues': '#fd7e14'})
           st.plotly_chart(fig, use_container_width=True)
       
       # Performance trends
       fig = go.Figure()
       
       fig.add_trace(go.Scatter(
           x=df['Verification'],
           y=df['Execution Time'],
           mode='lines+markers',
           name='Execution Time',
           line=dict(color='#2E86AB')
       ))
       
       fig.update_layout(
           title="Performance Trend",
           xaxis_title="Verification Number",
           yaxis_title="Execution Time (seconds)",
           height=400
       )
       
       st.plotly_chart(fig, use_container_width=True)
       
       # Summary statistics
       st.subheader("📊 Analytics Summary")
       
       col1, col2, col3, col4 = st.columns(4)
       
       with col1:
           avg_score = df['Overall Score'].mean()
           st.metric("Average Score", f"{avg_score:.1%}")
       
       with col2:
           total_issues = df['Total Issues'].sum()
           st.metric("Total Issues Found", total_issues)
       
       with col3:
           avg_time = df['Execution Time'].mean()
           st.metric("Avg Execution Time", f"{avg_time:.3f}s")
       
       with col4:
           improvement = df['Overall Score'].iloc[-1] - df['Overall Score'].iloc[0]
           st.metric("Score Improvement", f"{improvement:+.1%}")
