# ui/components/feedback_collector.py
"""
Feedback Collector Component - Collect user feedback for continuous improvement.
"""

import streamlit as st
from typing import Dict, Any
import json
import time

class FeedbackCollectorComponent:
   """Component for collecting user feedback on verification results"""
   
   def __init__(self):
       if 'feedback_data' not in st.session_state:
           st.session_state.feedback_data = []
   
   def render_feedback_form(self, result):
       """Render feedback collection form"""
       
       st.subheader("📝 Help Us Improve")
       st.write("Your feedback helps improve CodeX-Verify's accuracy and usefulness.")
       
       with st.form("feedback_form"):
           # Overall satisfaction
           satisfaction = st.select_slider(
               "How satisfied are you with the verification results?",
               options=["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"],
               value="Satisfied"
           )
           
           # Accuracy assessment
           accuracy = st.select_slider(
               "How accurate were the identified issues?",
               options=["Very Inaccurate", "Inaccurate", "Somewhat Accurate", "Accurate", "Very Accurate"],
               value="Accurate"
           )
           
           # Usefulness of suggestions
           usefulness = st.select_slider(
               "How useful were the improvement suggestions?",
               options=["Not Useful", "Slightly Useful", "Moderately Useful", "Useful", "Very Useful"],
               value="Useful"
           )
           
           # False positives/negatives
           col1, col2 = st.columns(2)
           
           with col1:
               false_positives = st.number_input(
                   "Number of false positives (issues incorrectly flagged):",
                   min_value=0,
                   max_value=len(result.aggregated_issues),
                   value=0
               )
           
           with col2:
               false_negatives = st.number_input(
                   "Estimated false negatives (real issues missed):",
                   min_value=0,
                   value=0
               )
           
           # Specific feedback
           specific_feedback = st.text_area(
               "Specific feedback or suggestions:",
               placeholder="Please describe any specific issues with the verification or suggestions for improvement..."
           )
           
           # Agent-specific feedback
           st.write("**Agent-Specific Feedback:**")
           agent_feedback = {}
           
           for agent_name in result.agent_results.keys():
               agent_rating = st.select_slider(
                   f"{agent_name.title()} Agent Performance:",
                   options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
                   value="Good",
                   key=f"agent_{agent_name}"
               )
               agent_feedback[agent_name] = agent_rating
           
           # Submit feedback
           submitted = st.form_submit_button("📤 Submit Feedback", type="primary")
           
           if submitted:
               self._save_feedback({
                   'timestamp': time.time(),
                   'result_summary': {
                       'overall_score': result.overall_score,
                       'total_issues': len(result.aggregated_issues),
                       'execution_time': result.execution_time
                   },
                   'satisfaction': satisfaction,
                   'accuracy': accuracy,
                   'usefulness': usefulness,
                   'false_positives': false_positives,
                   'false_negatives': false_negatives,
                   'specific_feedback': specific_feedback,
                   'agent_feedback': agent_feedback
               })
               
               st.success("🙏 Thank you for your feedback! This helps improve CodeX-Verify.")
   
   def _save_feedback(self, feedback_data: Dict[str, Any]):
       """Save feedback data to session state"""
       st.session_state.feedback_data.append(feedback_data)
   
   def render_feedback_analytics(self):
       """Render analytics from collected feedback"""
       
       if not st.session_state.feedback_data:
           st.info("No feedback data available yet")
           return
       
       st.subheader("📊 Feedback Analytics")
       
       feedback_df = pd.DataFrame(st.session_state.feedback_data)
       
       # Satisfaction metrics
       col1, col2, col3 = st.columns(3)
       
       with col1:
           avg_satisfaction = self._rating_to_numeric(feedback_df['satisfaction'].tolist())
           st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/5.0")
       
       with col2:
           avg_accuracy = self._rating_to_numeric(feedback_df['accuracy'].tolist())
           st.metric("Avg Accuracy", f"{avg_accuracy:.1f}/5.0")
       
       with col3:
           total_feedback = len(feedback_df)
           st.metric("Total Feedback", total_feedback)
       
       # False positive/negative trends
       if 'false_positives' in feedback_df.columns:
           fig = px.scatter(
               feedback_df.reset_index(),
               x='index',
               y='false_positives',
               title="False Positives Over Time",
               labels={'index': 'Feedback Number', 'false_positives': 'False Positives'}
           )
           st.plotly_chart(fig, use_container_width=True)
   
   def _rating_to_numeric(self, ratings: List[str]) -> float:
       """Convert text ratings to numeric values"""
       rating_map = {
           "Very Dissatisfied": 1, "Very Inaccurate": 1, "Not Useful": 1, "Poor": 1,
           "Dissatisfied": 2, "Inaccurate": 2, "Slightly Useful": 2, "Fair": 2,
           "Neutral": 3, "Somewhat Accurate": 3, "Moderately Useful": 3, "Good": 3,
           "Satisfied": 4, "Accurate": 4, "Useful": 4, "Very Good": 4,
           "Very Satisfied": 5, "Very Accurate": 5, "Very Useful": 5, "Excellent": 5
       }
       
       numeric_values = [rating_map.get(rating, 3) for rating in ratings]
       return sum(numeric_values) / len(numeric_values) if numeric_values else 0