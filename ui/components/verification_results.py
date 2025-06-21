# ui/components/verification_results.py
"""
Verification Results Component - Rich display of verification results.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from agents.base_agent import Severity

class VerificationResultsComponent:
   """Component for displaying detailed verification results"""
   
   def render(self, result):
       """Render main verification results"""
       
       if not result.aggregated_issues:
           st.success("🎉 Excellent! No issues detected across all verification agents.")
           self._render_success_metrics(result)
           return
       
       # Group issues by severity
       issues_by_severity = {
           Severity.CRITICAL: [],
           Severity.HIGH: [],
           Severity.MEDIUM: [],
           Severity.LOW: []
       }
       
       for issue in result.aggregated_issues:
           issues_by_severity[issue.severity].append(issue)
       
       # Render issues by severity with color coding
       severity_config = {
           Severity.CRITICAL: ("🚨", "error", "Critical Issues (Production Blockers)"),
           Severity.HIGH: ("⚠️", "warning", "High Priority Issues"),
           Severity.MEDIUM: ("ℹ️", "info", "Medium Priority Issues"),
           Severity.LOW: ("💡", "success", "Low Priority Suggestions")
       }
       
       for severity, (icon, alert_type, title) in severity_config.items():
           issues = issues_by_severity[severity]
           if issues:
               with st.expander(f"{icon} {title} ({len(issues)})", expanded=(severity in [Severity.CRITICAL, Severity.HIGH])):
                   self._render_issues_list(issues, alert_type)
   
   def _render_issues_list(self, issues: List, alert_type: str):
       """Render list of issues with consistent formatting"""
       for i, issue in enumerate(issues, 1):
           
           # Issue header with agent source
           agent_badge = f"**{getattr(issue, 'agent_source', 'Unknown').title()}**" if hasattr(issue, 'agent_source') else ""
           location = f" (Line {issue.line_number})" if issue.line_number else ""
           
           st.markdown(f"**{i}. {issue.type.replace('_', ' ').title()}**{location} {agent_badge}")
           
           # Issue details
           with st.container():
               st.write(f"📝 **Message:** {issue.message}")
               
               if issue.suggestion:
                   st.write(f"💡 **Suggestion:** {issue.suggestion}")
               
               # Additional metadata
               col1, col2, col3 = st.columns(3)
               with col1:
                   st.write(f"**Severity:** {issue.severity.value.title()}")
               with col2:
                   confidence = getattr(issue, 'confidence', 1.0)
                   st.write(f"**Confidence:** {confidence:.1%}")
               with col3:
                   if hasattr(issue, 'agent_source'):
                       st.write(f"**Source:** {issue.agent_source}")
               
               st.divider()
   
   def _render_success_metrics(self, result):
       """Render success metrics when no issues found"""
       col1, col2, col3, col4 = st.columns(4)
       
       with col1:
           st.metric("🧠 Correctness", "✅ Perfect")
       with col2:
           st.metric("🛡️ Security", "✅ Secure")
       with col3:
           st.metric("⚡ Performance", "✅ Optimized")
       with col4:
           st.metric("📝 Quality", "✅ Maintainable")
       
       # Show agent scores
       st.subheader("Agent Performance")
       agent_data = []
       for agent_name, agent_result in result.agent_results.items():
           agent_data.append({
               'Agent': agent_name.title(),
               'Score': agent_result.overall_score,
               'Status': '✅ Pass' if agent_result.overall_score >= 0.8 else '⚠️ Warning'
           })
       
       df = pd.DataFrame(agent_data)
       fig = px.bar(df, x='Agent', y='Score', 
                   title="Agent Verification Scores",
                   color='Score',
                   color_continuous_scale='RdYlGn',
                   range_y=[0, 1])
       st.plotly_chart(fig, use_container_width=True)
   
   def render_detailed_analysis(self, result):
       """Render detailed analysis view"""
       
       # Summary cards
       col1, col2, col3 = st.columns(3)
       
       with col1:
           st.markdown("""
           <div class="metric-card">
               <h3>🎯 Overall Assessment</h3>
               <p><strong>Score:</strong> {:.1%}</p>
               <p><strong>Grade:</strong> {}</p>
               <p><strong>Status:</strong> {}</p>
           </div>
           """.format(
               result.overall_score,
               result.summary.get('grade', 'N/A'),
               result.overall_status
           ), unsafe_allow_html=True)
       
       with col2:
           critical_count = len([i for i in result.aggregated_issues if i.severity == Severity.CRITICAL])
           high_count = len([i for i in result.aggregated_issues if i.severity == Severity.HIGH])
           
           st.markdown("""
           <div class="metric-card">
               <h3>🚨 Risk Assessment</h3>
               <p><strong>Critical:</strong> {}</p>
               <p><strong>High:</strong> {}</p>
               <p><strong>Total Issues:</strong> {}</p>
           </div>
           """.format(critical_count, high_count, len(result.aggregated_issues)), unsafe_allow_html=True)
       
       with col3:
           enterprise_metrics = result.metadata.get('enterprise_metrics', {})
           deployment_risk = enterprise_metrics.get('deployment_risk', 'UNKNOWN')
           
           st.markdown("""
           <div class="metric-card">
               <h3>🏢 Enterprise Readiness</h3>
               <p><strong>Deployment Risk:</strong> {}</p>
               <p><strong>Production Ready:</strong> {}</p>
               <p><strong>Security Posture:</strong> {:.1%}</p>
           </div>
           """.format(
               deployment_risk,
               "Yes" if result.overall_score >= 0.85 else "No",
               enterprise_metrics.get('security_posture', 0)
           ), unsafe_allow_html=True)
       
       # Agent breakdown
       st.subheader("🤖 Agent Analysis Breakdown")
       
       agent_tabs = st.tabs([name.title() for name in result.agent_results.keys()])
       
       for tab, (agent_name, agent_result) in zip(agent_tabs, result.agent_results.items()):
           with tab:
               self._render_agent_detailed_analysis(agent_name, agent_result)
   
   def _render_agent_detailed_analysis(self, agent_name: str, agent_result):
       """Render detailed analysis for specific agent"""
       
       # Agent performance summary
       col1, col2, col3, col4 = st.columns(4)
       
       with col1:
           st.metric("Score", f"{agent_result.overall_score:.1%}")
       with col2:
           st.metric("Issues Found", len(agent_result.issues))
       with col3:
           st.metric("Execution Time", f"{agent_result.execution_time:.3f}s")
       with col4:
           status = "✅ Success" if agent_result.success else "❌ Failed"
           st.metric("Status", status)
       
       # Agent-specific issues
       if agent_result.issues:
           st.subheader(f"Issues Detected by {agent_name.title()}")
           self._render_issues_list(agent_result.issues, "info")
       else:
           st.success(f"🎉 {agent_name.title()} agent found no issues!")
       
       # Agent metadata
       if agent_result.metadata:
           with st.expander("🔧 Agent Metadata"):
               st.json(agent_result.metadata)
