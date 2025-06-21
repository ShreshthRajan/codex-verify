# ui/components/code_editor.py
"""
Code Editor Component - Enhanced code input with syntax highlighting and validation.
"""

import streamlit as st
from typing import Optional, Dict, Any
import re

class CodeEditorComponent:
   """Component for enhanced code editing and input"""
   
   def __init__(self):
       self.example_codes = self._load_example_codes()
   
   def render_enhanced_editor(self, initial_code: str = "") -> str:
       """Render enhanced code editor with features"""
       
       # Editor tabs
       editor_tab1, editor_tab2, editor_tab3 = st.tabs(["✍️ Editor", "📚 Examples", "📋 Templates"])
       
       with editor_tab1:
           code = self._render_main_editor(initial_code)
       
       with editor_tab2:
           example_code = self._render_examples()
           if example_code:
               code = example_code
       
       with editor_tab3:
           template_code = self._render_templates()
           if template_code:
               code = template_code
       
       return code
   
   def _render_main_editor(self, initial_code: str) -> str:
       """Render main code editor"""
       
       # Editor settings
       col1, col2, col3 = st.columns(3)
       
       with col1:
           line_numbers = st.checkbox("Show line numbers", value=True)
       
       with col2:
           syntax_highlighting = st.checkbox("Syntax highlighting", value=True)
       
       with col3:
           word_wrap = st.checkbox("Word wrap", value=False)
       
       # Main code editor
       code = st.text_area(
           "Python Code:",
           value=initial_code,
           height=400,
           help="Enter your Python code here. The editor supports syntax highlighting and line numbers."
       )
       
       # Live code analysis
       if code.strip():
           self._render_live_analysis(code)
       
       return code
   
   def _render_live_analysis(self, code: str):
       """Render live code analysis while typing"""
       
       st.subheader("🔍 Live Analysis")
       
       # Basic code metrics
       lines = code.splitlines()
       non_empty_lines = [line for line in lines if line.strip()]
       
       col1, col2, col3, col4 = st.columns(4)
       
       with col1:
           st.metric("Total Lines", len(lines))
       
       with col2:
           st.metric("Code Lines", len(non_empty_lines))
       
       with col3:
           comment_lines = len([line for line in lines if line.strip().startswith('#')])
           st.metric("Comments", comment_lines)
       
       with col4:
           # Simple complexity estimate
           complexity_indicators = code.count('if') + code.count('for') + code.count('while') + code.count('try')
           complexity = "Low" if complexity_indicators < 5 else "Medium" if complexity_indicators < 15 else "High"
           st.metric("Complexity", complexity)
       
       # Quick syntax check
       try:
           compile(code, '<string>', 'exec')
           st.success("✅ Syntax is valid")
       except SyntaxError as e:
           st.error(f"❌ Syntax Error: {e.msg} (Line {e.lineno})")
       except Exception as e:
           st.warning(f"⚠️ Compilation Warning: {str(e)}")
       
       # Quick security scan
       security_patterns = [
           (r'import os', 'OS operations detected'),
           (r'subprocess', 'Subprocess usage detected'),
           (r'eval\s*\(', 'eval() usage detected - security risk'),
           (r'exec\s*\(', 'exec() usage detected - security risk'),
           (r'pickle\.loads', 'pickle.loads() detected - security risk')
       ]
       
       security_issues = []
       for pattern, message in security_patterns:
           if re.search(pattern, code):
               security_issues.append(message)
       
       if security_issues:
           with st.expander("⚠️ Quick Security Scan"):
               for issue in security_issues:
                   st.warning(f"🔍 {issue}")
   
   def _render_examples(self) -> Optional[str]:
       """Render code examples"""
       
       st.subheader("📚 Code Examples")
       
       example_category = st.selectbox(
           "Choose example category:",
           ["Basic Python", "Data Processing", "Web Development", "Machine Learning", "Security Testing"]
       )
       
       examples = self.example_codes.get(example_category, {})
       
       if examples:
           example_name = st.selectbox("Choose example:", list(examples.keys()))
           
           if example_name:
               example_code = examples[example_name]
               
               st.code(example_code, language='python')
               
               if st.button(f"Use {example_name}", key=f"use_example_{example_name}"):
                   st.session_state.selected_example = example_code
                   return example_code
       
       return None
   
   def _render_templates(self) -> Optional[str]:
       """Render code templates"""
       
       st.subheader("📋 Code Templates")
       
       templates = {
           "Function Template": '''def function_name(parameter1, parameter2):
   """
   Brief description of the function.
   
   Args:
       parameter1: Description of parameter1
       parameter2: Description of parameter2
   
   Returns:
       Description of return value
   """
   # Implementation here
   return result''',
           
           "Class Template": '''class ClassName:
   """Brief description of the class."""
   
   def __init__(self, parameter):
       """Initialize the class."""
       self.parameter = parameter
   
   def method_name(self):
       """Brief description of the method."""
       pass''',
           
           "Error Handling Template": '''try:
   # Code that might raise an exception
   result = risky_operation()
   
except SpecificException as e:
   # Handle specific exception
   print(f"Specific error occurred: {e}")
   
except Exception as e:
   # Handle general exceptions
   print(f"Unexpected error: {e}")
   
else:
   # Code to run if no exception occurred
   print("Operation successful")
   
finally:
   # Code that always runs
   cleanup_resources()''',
           
           "Data Processing Template": '''import pandas as pd

def process_data(input_file):
   """
   Process data from input file.
   
   Args:
       input_file: Path to input CSV file
   
   Returns:
       Processed DataFrame
   """
   try:
       # Load data
       df = pd.read_csv(input_file)
       
       # Data validation
       if df.empty:
           raise ValueError("Input file is empty")
       
       # Data processing
       processed_df = df.dropna()  # Remove missing values
       processed_df = processed_df.drop_duplicates()  # Remove duplicates
       
       return processed_df
       
   except FileNotFoundError:
       print(f"Error: File {input_file} not found")
       return None
   
   except Exception as e:
       print(f"Error processing data: {e}")
       return None''',
           
           "API Template": '''import requests
import json

class APIClient:
   """Simple API client template."""
   
   def __init__(self, base_url, api_key=None):
       self.base_url = base_url
       self.api_key = api_key
       self.session = requests.Session()
       
       if api_key:
           self.session.headers.update({
               'Authorization': f'Bearer {api_key}'
           })
   
   def get(self, endpoint, params=None):
       """Make GET request to API."""
       try:
           url = f"{self.base_url}/{endpoint}"
           response = self.session.get(url, params=params)
           response.raise_for_status()
           return response.json()
           
       except requests.exceptions.RequestException as e:
           print(f"API request failed: {e}")
           return None
   
   def post(self, endpoint, data=None):
       """Make POST request to API."""
       try:
           url = f"{self.base_url}/{endpoint}"
           response = self.session.post(url, json=data)
           response.raise_for_status()
           return response.json()
           
       except requests.exceptions.RequestException as e:
           print(f"API request failed: {e}")
           return None'''
       }
       
       template_name = st.selectbox("Choose template:", list(templates.keys()))
       
       if template_name:
           template_code = templates[template_name]
           
           st.code(template_code, language='python')
           
           if st.button(f"Use {template_name}", key=f"use_template_{template_name}"):
               return template_code
       
       return None
   
   def _load_example_codes(self) -> Dict[str, Dict[str, str]]:
       """Load example codes by category"""
       
       return {
           "Basic Python": {
               "Hello World": '''def hello_world(name="World"):
   """Simple greeting function."""
   return f"Hello, {name}!"

if __name__ == "__main__":
   print(hello_world())
   print(hello_world("CodeX-Verify"))''',
               
               "List Processing": '''def process_numbers(numbers):
   """Process a list of numbers with various operations."""
   if not numbers:
       return {}
   
   return {
       'sum': sum(numbers),
       'average': sum(numbers) / len(numbers),
       'max': max(numbers),
       'min': min(numbers),
       'even_count': len([n for n in numbers if n % 2 == 0])
   }

# Example usage
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = process_numbers(numbers)
print(result)''',
               
               "File Operations": '''import os

def read_and_process_file(filename):
   """Read a text file and return word count."""
   try:
       with open(filename, 'r', encoding='utf-8') as file:
           content = file.read()
           
       words = content.split()
       return {
           'total_words': len(words),
           'unique_words': len(set(words)),
           'characters': len(content),
           'lines': len(content.splitlines())
       }
       
   except FileNotFoundError:
       return {"error": f"File {filename} not found"}
   except Exception as e:
       return {"error": str(e)}'''
           },
           
           "Data Processing": {
               "CSV Analysis": '''import pandas as pd
import numpy as np

def analyze_csv(file_path):
   """Analyze a CSV file and return statistics."""
   try:
       df = pd.read_csv(file_path)
       
       analysis = {
           'shape': df.shape,
           'columns': df.columns.tolist(),
           'missing_values': df.isnull().sum().to_dict(),
           'data_types': df.dtypes.to_dict(),
           'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
       }
       
       return analysis
       
   except Exception as e:
       return {"error": str(e)}

# Example usage
# analysis = analyze_csv('data.csv')
# print(analysis)''',
               
               "Data Cleaning": '''import pandas as pd

def clean_dataset(df):
   """Clean a pandas DataFrame."""
   if df.empty:
       return df
   
   # Create a copy to avoid modifying original
   cleaned_df = df.copy()
   
   # Remove duplicates
   cleaned_df = cleaned_df.drop_duplicates()
   
   # Handle missing values
   for column in cleaned_df.columns:
       if cleaned_df[column].dtype == 'object':
           # Fill categorical missing values with mode
           mode_value = cleaned_df[column].mode()
           if not mode_value.empty:
               cleaned_df[column].fillna(mode_value[0], inplace=True)
       else:
           # Fill numeric missing values with median
           cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
   
   return cleaned_df'''
           },
           
           "Security Testing": {
               "Input Validation": '''def validate_email(email):
   """Validate email format with security considerations."""
   import re
   
   if not email or not isinstance(email, str):
       return False
   
   # Length check to prevent DoS
   if len(email) > 254:
       return False
   
   # Basic email pattern
   pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
   
   return re.match(pattern, email) is not None

def sanitize_input(user_input):
   """Sanitize user input to prevent injection attacks."""
   if not isinstance(user_input, str):
       return str(user_input)
   
   # Remove potentially dangerous characters
   dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`']
   
   sanitized = user_input
   for char in dangerous_chars:
       sanitized = sanitized.replace(char, '')
   
   return sanitized.strip()''',
               
               "Password Security": '''import hashlib
import secrets
import string

def generate_secure_password(length=12):
   """Generate a cryptographically secure password."""
   if length < 8:
       raise ValueError("Password length should be at least 8 characters")
   
   alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
   password = ''.join(secrets.choice(alphabet) for _ in range(length))
   
   return password

def hash_password(password, salt=None):
   """Hash password using SHA-256 with salt."""
   if salt is None:
       salt = secrets.token_hex(32)
   
   # Combine password and salt
   password_salt = password + salt
   
   # Hash using SHA-256
   hashed = hashlib.sha256(password_salt.encode()).hexdigest()
   
   return {
       'hash': hashed,
       'salt': salt
   }

def verify_password(password, stored_hash, salt):
   """Verify password against stored hash."""
   computed_hash = hash_password(password, salt)['hash']
   return computed_hash == stored_hash'''
           }
       }
