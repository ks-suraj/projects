import os
import sys
import ast
import glob
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import log_info, log_error
from utils.api_client import call_openrouter_api

class CodeFrameworkAnalyzer:
    FRAMEWORK_PATTERNS = {
        'pytorch': ['torch', 'nn.Module', 'optim'],
        'tensorflow': ['tensorflow', 'keras', 'tf.'],
        'django': ['django', 'models.Model', 'views'],
        'flask': ['flask', 'Flask', 'request'],
        'fastapi': ['fastapi', 'APIRouter', 'Depends'],
        'matplotlib': ['matplotlib', 'pyplot', 'plt'],
        'pygame': ['pygame', 'Surface', 'sprite'],
        'pandas': ['pandas', 'DataFrame', 'Series'],
        'numpy': ['numpy', 'ndarray', 'array'],
        'scikit-learn': ['sklearn', 'fit', 'predict'],
        'selenium': ['webdriver', 'By', 'WebElement'],
        'beautifulsoup': ['BeautifulSoup', 'bs4'],
        'requests': ['requests', 'Response'],
        'sqlalchemy': ['sqlalchemy', 'Column', 'Model'],
        'pydantic': ['pydantic', 'BaseModel'],
        'xgboost': ['xgboost', 'DMatrix', 'train'],
        'lightgbm': ['lightgbm', 'LGBMClassifier', 'LGBMRegressor'],
        'transformers': ['transformers', 'AutoModel', 'AutoTokenizer'],
        'nltk': ['nltk', 'word_tokenize', 'pos_tag'],
        'spacy': ['spacy', 'Language', 'Doc'],
        'opencv': ['cv2', 'imread', 'imshow'],
        'gymnasium': ['gymnasium', 'make', 'Env'],
        'stable_baselines3': ['stable_baselines3', 'PPO', 'A2C'],
        'torch_geometric': ['torch_geometric', 'Data', 'GCNConv'],
        'cvxpy': ['cvxpy', 'Variable', 'Problem'],
        'statsmodels': ['statsmodels', 'OLS', 'ARIMA'],
        'simpy': ['simpy', 'Environment', 'Process'],
        'sympy': ['sympy', 'symbols', 'solve'],
        'qiskit': ['qiskit', 'QuantumCircuit', 'Aer'],
        'geopandas': ['geopandas', 'GeoDataFrame', 'read_file'],
        'control': ['control', 'tf', 'step_response'],
        'biopython': ['Bio', 'Seq', 'SeqIO']
    }
    
    def detect_frameworks(self, code: str) -> List[str]:
        frameworks = []
        try:
            for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                if any(pattern in code for pattern in patterns):
                    frameworks.append(framework)
        except Exception as e:
            log_error(f"Framework detection error: {e}")
        return frameworks

class CodeQualityChecker:
    def __init__(self):
        self.checks = {
            'general': self._check_general_issues,
            'documentation': self._check_documentation,
            'error_handling': self._check_error_handling,
            'performance': self._check_performance,
            'security': self._check_security,
            'best_practices': self._check_best_practices
        }

    def _check_general_issues(self, ast_tree: ast.AST) -> List[str]:
        issues = []
        try:
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.ClassDef):
                    self._check_class_structure(node, issues)
                elif isinstance(node, ast.FunctionDef):
                    self._check_function_structure(node, issues)
        except Exception as e:
            log_error(f"General check error: {e}")
        return issues

    def _check_class_structure(self, node: ast.ClassDef, issues: List[str]) -> None:
        try:
            has_init = any(isinstance(n, ast.FunctionDef) and n.name == '__init__' 
                          for n in node.body)
            if not has_init:
                issues.append(f"Class {node.name} missing __init__ method")

            if not ast.get_docstring(node):
                issues.append(f"Class {node.name} missing docstring")

            for method in (n for n in node.body if isinstance(n, ast.FunctionDef)):
                self._check_function_structure(method, issues, class_name=node.name)
        except Exception as e:
            log_error(f"Class structure check error: {e}")

    def _check_function_structure(self, node: ast.FunctionDef, issues: List[str], 
                                class_name: str = None) -> None:
        try:
            func_name = f"{class_name}.{node.name}" if class_name else node.name
            
            if hasattr(node, 'returns') and not node.returns and node.name != '__init__':
                issues.append(f"Function {func_name} missing return type hint")
            
            if hasattr(node, 'args'):
                for arg in node.args.args[1:]:  # Skip 'self' for methods
                    if hasattr(arg, 'annotation') and not arg.annotation:
                        issues.append(f"Argument {arg.arg} in {func_name} missing type hint")
        except Exception as e:
            log_error(f"Function structure check error: {e}")

    def _check_documentation(self, ast_tree: ast.AST) -> List[str]:
        issues = []
        try:
            for node in ast.walk(ast_tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    if not ast.get_docstring(node):
                        name = getattr(node, 'name', 'module')
                        issues.append(f"{node.__class__.__name__} {name} missing docstring")
        except Exception as e:
            log_error(f"Documentation check error: {e}")
        return issues

    def _check_error_handling(self, ast_tree: ast.AST) -> List[str]:
        # NOTE: AST nodes do not have parent pointers, so this check is limited.
        # For a more robust solution, use astroid or a custom parent-tracking traversal.
        issues = []
        try:
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.Call):
                    # This is a placeholder: real error handling detection is non-trivial.
                    pass
        except Exception as e:
            log_error(f"Error handling check error: {e}")
        return issues

    def _check_performance(self, ast_tree: ast.AST) -> List[str]:
        issues = []
        try:
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.For):
                    if any(isinstance(n, ast.Call) for n in ast.walk(node)):
                        issues.append("Consider using list comprehension or vectorized operations")
        except Exception as e:
            log_error(f"Performance check error: {e}")
        return issues

    def _check_security(self, ast_tree: ast.AST) -> List[str]:
        issues = []
        try:
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.Call):
                    func_name = ''
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                    
                    if func_name in ['eval', 'exec', 'input', 'pickle.loads']:
                        issues.append(f"Unsafe {func_name} usage detected")
        except Exception as e:
            log_error(f"Security check error: {e}")
        return issues

    def _check_best_practices(self, ast_tree: ast.AST) -> List[str]:
        issues = []
        try:
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.Import):
                    if any(name.name == '*' for name in node.names):
                        issues.append("Wildcard imports are not recommended")
                elif isinstance(node, ast.Compare):
                    if isinstance(node.ops[0], (ast.Is, ast.IsNot)):
                        issues.append("Use == instead of 'is' for value comparison")
        except Exception as e:
            log_error(f"Best practices check error: {e}")
        return issues

class CodeExpertAgent:
    def __init__(self):
        self.framework_analyzer = CodeFrameworkAnalyzer()
        self.quality_checker = CodeQualityChecker()
        self.codes_dir = "d:/ai/projects/Real-World-Projects/genesis-self-evolving-ai/codes"
        self.output_dir = "d:/ai/projects/Real-World-Projects/genesis-self-evolving-ai/codes/improved"
        os.makedirs(self.output_dir, exist_ok=True)

    def get_latest_code(self) -> str:
        try:
            code_files = glob.glob(os.path.join(self.codes_dir, "*.py"))
            if not code_files:
                raise FileNotFoundError("No code files found")
            return max(code_files, key=os.path.getctime)
        except Exception as e:
            log_error(f"Failed to get latest code: {e}")
            raise

    def clean_code_content(self, content: str) -> str:
        try:
            lines = content.split('\n')
            cleaned_lines = []
            for line in lines:
                if not line.strip().startswith('```') and not line.endswith('```'):
                    cleaned_lines.append(line)
            return '\n'.join(cleaned_lines)
        except Exception as e:
            log_error(f"Code cleaning error: {e}")
            return content

    def analyze_code(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            code = self.clean_code_content(content)
            
            try:
                ast_tree = ast.parse(code)
            except SyntaxError as se:
                return self._handle_syntax_error(se, code)
            
            frameworks = self.framework_analyzer.detect_frameworks(code)
            
            analysis = {
                'frameworks': frameworks,
                'issues': [],
                'code': code
            }
            
            for check_name, check_func in self.quality_checker.checks.items():
                try:
                    issues = check_func(ast_tree)
                    if issues:
                        analysis['issues'].extend(issues)
                except Exception as e:
                    log_error(f"Error in {check_name} check: {e}")
                    
            return analysis
            
        except Exception as e:
            log_error(f"Failed to analyze file: {e}")
            return {'error': str(e)}

    def _handle_syntax_error(self, se: SyntaxError, code: str) -> Dict[str, Any]:
        line_no = se.lineno if hasattr(se, 'lineno') else 'unknown'
        col_no = se.offset if hasattr(se, 'offset') else 'unknown'
        error_line = se.text if hasattr(se, 'text') else 'unknown'
        
        detailed_error = f"Syntax error at line {line_no}, column {col_no}\n"
        detailed_error += f"Error details: {str(se)}\n"
        detailed_error += f"Problematic line: {error_line}"
        
        log_error(detailed_error)
        return {
            'error': detailed_error,
            'code': code,
            'line_number': line_no
        }

    def generate_improvement_prompt(self, analysis: Dict[str, Any]) -> str:
        frameworks = ', '.join(analysis['frameworks']) if analysis['frameworks'] else 'general Python'
        
        prompt = f"""Improve this {frameworks} code for production use. Make it flawless by addressing:

1. Code Issues Found:
{chr(10).join(f'- {issue}' for issue in analysis['issues'])}

2. Framework-Specific Best Practices:
- Follow {frameworks} conventions and patterns
- Implement proper error handling
- Add comprehensive documentation
- Optimize performance
- Ensure type safety
- Add proper logging
- Implement testing hooks

Original code:
{analysis['code']}

Return only the improved code without explanations or markdown. STRICTLY GIVE ONLY PYTHON CODE, NO TEXT FORM"""
        
        return prompt

    def improve_code(self, analysis: Dict[str, Any]) -> Optional[str]:
        try:
            if len(analysis['code']) > 4000:
                return self._improve_large_code(analysis)
            return self._improve_small_code(analysis)
        except Exception as e:
            log_error(f"Failed to improve code: {e}")
            return None

    def _improve_large_code(self, analysis: Dict[str, Any]) -> Optional[str]:
        try:
            code_chunks = [analysis['code'][i:i+4000] 
                         for i in range(0, len(analysis['code']), 4000)]
            improved_chunks = []
            
            for i, chunk in enumerate(code_chunks):
                chunk_prompt = self._generate_chunk_prompt(chunk, analysis, i, len(code_chunks))
                response = self._call_api_with_retry(chunk_prompt)
                if response:
                    improved_chunks.append(response)
                    
            return '\n'.join(improved_chunks) if improved_chunks else None
        except Exception as e:
            log_error(f"Large code improvement failed: {e}")
            return None

    def _improve_small_code(self, analysis: Dict[str, Any]) -> Optional[str]:
        try:
            prompt = self.generate_improvement_prompt(analysis)
            return self._call_api_with_retry(prompt)
        except Exception as e:
            log_error(f"Small code improvement failed: {e}")
            return None

    def _generate_chunk_prompt(self, chunk: str, analysis: Dict[str, Any], 
                             chunk_index: int, total_chunks: int) -> str:
        return f"""Improve this code chunk ({chunk_index + 1}/{total_chunks}):

{chunk}

Focus on:
1. Maintaining consistency with other chunks
2. Adding proper documentation
3. Implementing error handling
4. Following {', '.join(analysis['frameworks'])} best practices

Return only the improved code without explanations or markdown."""

    def _call_api_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        for attempt in range(max_retries):
            try:
                response = call_openrouter_api([{"role": "user", "content": prompt}])
                return response
            except Exception as e:
                log_error(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def save_improved_code(self, code: str, original_file: str) -> Optional[str]:
        if not code:
            return None
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.basename(original_file)
            improved_name = f"improved_{timestamp}_{base_name}"
            output_path = os.path.join(self.output_dir, improved_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            log_info(f"Saved improved code to {output_path}")
            return output_path
        except Exception as e:
            log_error(f"Failed to save improved code: {e}")
            return None

def run_code_expert() -> bool:
    try:
        log_info("Starting Code Expert Agent...")
        agent = CodeExpertAgent()
        
        latest_code = agent.get_latest_code()
        log_info(f"Analyzing code: {latest_code}")
        
        analysis = agent.analyze_code(latest_code)
        if 'error' in analysis:
            log_error(f"Analysis failed: {analysis['error']}")
            return False
            
        frameworks = analysis.get('frameworks', [])
        log_info(f"Detected frameworks: {', '.join(frameworks)}")
        
        if analysis['issues']:
            log_info(f"Found {len(analysis['issues'])} issues to address")
            improved_code = agent.improve_code(analysis)
            if improved_code:
                output_path = agent.save_improved_code(improved_code, latest_code)
                if output_path:
                    log_info("Code improvement completed successfully")
                    return True
        else:
            log_info("No issues found in the code")
            return True
            
    except Exception as e:
        log_error(f"Code Expert Agent failed: {e}")
        return False

if __name__ == "__main__":
    success = run_code_expert()
    if success:
        log_info("Code Expert Agent completed successfully")
    else:
        log_error("Code Expert Agent encountered errors")