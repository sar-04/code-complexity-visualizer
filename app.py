import os
import re
import streamlit as st
from radon.visitors import ComplexityVisitor
from radon.metrics import mi_visit
import google.generativeai as genai
import ast

# 1) Page configuration
st.set_page_config(page_title="Code Complexity Visualizer", page_icon="🔬", layout="centered")

# 2) API Key Management
GEMINI_API_KEY = None
try:
	# Prefer Streamlit secrets when available
	if "GEMINI_API_KEY" in st.secrets:
		GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
	# Secrets not available (e.g., local execution)
	pass

# Fallback to environment variable if not found in secrets
if not GEMINI_API_KEY:
	GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
	genai.configure(api_key=GEMINI_API_KEY)
else:
	st.warning("No Gemini API key found. Add it to Streamlit secrets or set environment variable 'GEMINI_API_KEY'. AI refactoring will be disabled.")

# Utilities: sanitization and validation

def sanitize_code(code: str) -> str:
	"""Normalize newlines and strip zero-width/hidden characters that break parsers."""
	if not code:
		return ""
	clean = code.replace("\r\n", "\n").replace("\r", "\n")
	# Remove BOM and zero-width characters
	clean = clean.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
	# Ensure trailing newline to help some parsers
	if not clean.endswith("\n"):
		clean += "\n"
	return clean

def validate_python(code: str):
	"""Return (ok, err) where err includes line/col and message for SyntaxError."""
	try:
		ast.parse(code)
		return True, None
	except SyntaxError as e:
		line = e.lineno or 0
		col = e.offset or 0
		msg = f"SyntaxError at line {line}, column {col}: {e.msg}"
		return False, msg
	except Exception as e:
		return False, f"Parse error: {e}"

# 3) Core Analysis Function

def analyze_code_complexity(code: str):
	"""Return (total_cyclomatic_complexity, maintainability_index).
	Robust to Radon failures: falls back to AST-based CC and MI=0.0.
	"""
	def _approximate_cc_with_ast(src: str) -> int:
		try:
			tree = ast.parse(src)
		except Exception:
			return 0
		cc = 0
		for node in ast.walk(tree):
			if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try)):
				cc += 1
			elif isinstance(node, ast.BoolOp):
				# Count boolean operators (and/or) as additional decision points
				cc += max(0, len(getattr(node, 'values', [])) - 1)
			elif isinstance(node, (ast.comprehension,)):
				cc += 1
			elif isinstance(node, ast.IfExp):
				cc += 1
		return max(cc, 0)

	# Compute Cyclomatic Complexity (prefer Radon, fallback to AST)
	try:
		visitor = ComplexityVisitor.from_code(code)
		total_complexity = sum(
			block.complexity for block in (visitor.functions + visitor.methods)
		)
	except Exception:
		total_complexity = _approximate_cc_with_ast(code)

	# Compute Maintainability Index (best effort)
	maintainability_index = None
	try:
		single_mi = mi_visit(code, multi=False)
		maintainability_index = float(single_mi)
	except Exception:
		try:
			mi_scores = mi_visit(code, multi=True)
			values = []
			if isinstance(mi_scores, dict):
				for v in mi_scores.values():
					try:
						values.append(float(v))
					except Exception:
						pass
			elif isinstance(mi_scores, (list, tuple)):
				for item in mi_scores:
					if isinstance(item, (int, float)):
						values.append(float(item))
					elif (
						isinstance(item, (list, tuple))
						and len(item) >= 2
						and isinstance(item[1], (int, float))
					):
						values.append(float(item[1]))
			if values:
				maintainability_index = sum(values) / len(values)
		except Exception:
			maintainability_index = None

	if maintainability_index is None:
		maintainability_index = 0.0

	return total_complexity, maintainability_index

# Helper: extract per-function slices using AST

def extract_functions(code: str):
	"""Return list of dicts: {name, start, end, code} for top-level and nested functions."""
	results = []
	try:
		tree = ast.parse(code)
		lines = code.splitlines()
		for node in ast.walk(tree):
			if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
				name = node.name
				start = getattr(node, 'lineno', None)
				end = getattr(node, 'end_lineno', None)
				if start is not None and end is not None and 1 <= start <= end <= len(lines):
					snippet = "\n".join(lines[start-1:end])
					results.append({
						"name": name,
						"start": start,
						"end": end,
						"code": snippet,
					})
	except Exception:
		return []
	return results

# 4) Core LLM Refactoring Function

def _is_api_configured() -> bool:
	return bool(GEMINI_API_KEY)

def _strip_markdown_fences(text: str) -> str:
	# Remove triple backtick code fences, optionally with language
	text = re.sub(r"^\s*```[a-zA-Z0-9_+-]*\s*\n", "", text)
	text = re.sub(r"\n\s*```\s*$", "", text)
	# Also remove stray backticks
	text = text.replace("```python", "").replace("```", "").strip()
	return text

def get_llm_refactoring_suggestion(code: str, complexity_score: float) -> str:
	"""Return refactored code suggestion from Gemini, or error text if unavailable."""
	if not _is_api_configured():
		return "[Error] Gemini API key not configured."
	try:
		model = genai.GenerativeModel('gemini-1.5-flash')
		prompt = (
			"You are an expert Python developer. Your goal is to reduce cyclomatic complexity "
			"and improve maintainability while preserving behavior and I/O.\n\n"
			f"Given this Python code (cyclomatic complexity score: {complexity_score}), refactor it.\n"
			"Strict requirements:\n"
			"- Preserve functionality and public API.\n"
			"- Reduce branching and deeply nested logic; split into small functions where helpful.\n"
			"- Prefer clear names and straight-line logic; avoid cleverness.\n"
			"- Keep imports minimal and standard.\n"
			"- Only output the complete refactored Python code in a SINGLE code block.\n"
			"- Do NOT include any explanation before or after the code.\n\n"
			f"Original code:\n<CODE>\n{code}\n</CODE>\n\n"
			"Output: a single Python code block with the refactoring."
		)
		response = model.generate_content(prompt)
		text = getattr(response, 'text', '') or ''
		if not text:
			# Some SDK versions nest parts differently
			candidates = getattr(response, 'candidates', None)
			if candidates:
				text = candidates[0].content.parts[0].text if candidates[0].content.parts else ''
		suggestion = _strip_markdown_fences(text)
		return suggestion if suggestion.strip() else "[Error] Empty response from model."
	except Exception as e:
		return f"[Error] Failed to generate suggestion: {e}"

# 5) Streamlit User Interface
st.title("🔬 Code Complexity Visualizer")
st.markdown(
	"Analyze Python code for Cyclomatic Complexity and Maintainability Index. "
	"If code is overly complex, an AI refactoring suggestion will be generated using Google Gemini."
)

user_code = st.text_area(
	"Paste your Python function or module here:",
	height=280,
	placeholder="""def example(a, b):
	if a > 0:
		if b > 0:
			return a + b
		else:
			return a - b
	else:
		return b - a""",
)

analyze_clicked = st.button("Analyze Code")

if analyze_clicked:
	if not user_code or not user_code.strip():
		st.error("Please paste some Python code before analyzing.")
	else:
		cleaned = sanitize_code(user_code)
		ok, err = validate_python(cleaned)
		if not ok:
			st.error(f"Failed to analyze code: {err}")
		else:
			total_complexity, mi_score = analyze_code_complexity(cleaned)
			if total_complexity is None or mi_score is None:
				st.error("Failed to analyze code. Ensure the code is valid Python.")
			else:
				st.header("Analysis Results")
				col1, col2 = st.columns(2)

				with col1:
					st.metric(label="Cyclomatic Complexity (Σ of functions/methods)", value=int(total_complexity))
					if total_complexity < 6:
						st.success("Lower is better. 1-5 Good")
					elif 6 <= total_complexity <= 10:
						st.warning("Moderate complexity. 6-10 Moderate")
					else:
						st.error("High complexity. 11+ High")

				with col2:
					st.metric(label="Maintainability Index (0-100)", value=round(float(mi_score), 2))
					if mi_score > 19:
						st.success("Higher is better. 20-100 High")
					elif 10 <= mi_score <= 19:
						st.warning("Medium maintainability. 10-19 Medium")
					else:
						st.error("Low maintainability. 0-9 Low")

				# Per-function refactor workflow
				functions = extract_functions(cleaned)
				if functions:
					st.subheader("Per-function Refactor")
					# Compute per-function metrics
					items = []
					for fn in functions:
						cc, mi = analyze_code_complexity(fn["code"])
						if cc is not None and mi is not None:
							items.append({"label": f"{fn['name']} (L{fn['start']}-{fn['end']}) – CC {int(cc)}, MI {round(float(mi),2)}", "fn": fn, "cc": cc, "mi": mi})
					if items:
						labels = [it["label"] for it in items]
						choice = st.selectbox("Select a function to refactor", labels, index=0)
						selected = next((it for it in items if it["label"] == choice), None)
						can_refactor = _is_api_configured()
						refactor_clicked = st.button("Refactor selected function", disabled=not can_refactor)
						if not can_refactor:
							st.info("Provide GEMINI_API_KEY to enable AI refactoring.")
						if refactor_clicked and selected:
							with st.spinner("Generating function refactor..."):
								suggestion = get_llm_refactoring_suggestion(selected["fn"]["code"], selected["cc"])
							# Compute before/after metrics
							before_cc, before_mi = selected["cc"], selected["mi"]
							after_cc, after_mi = analyze_code_complexity(suggestion)
							col_a, col_b = st.columns(2)
							with col_a:
								st.subheader("Original Function")
								st.code(selected["fn"]["code"], language="python")
								st.metric("CC (before)", int(before_cc))
								st.metric("MI (before)", round(float(before_mi), 2))
							with col_b:
								st.subheader("Suggested Refactoring")
								st.code(suggestion, language="python")
								if after_cc is not None and after_mi is not None:
									st.metric("CC (after)", int(after_cc))
									st.metric("MI (after)", round(float(after_mi), 2))
									if int(after_cc) < int(before_cc) and float(after_mi) >= float(before_mi):
										st.success("Improved complexity and maintainability.")
									elif int(after_cc) < int(before_cc):
										st.success("Improved complexity.")
									elif float(after_mi) > float(before_mi):
										st.success("Improved maintainability.")
									else:
										st.warning("No improvement detected.")
					else:
						st.warning("No functions detected or metrics unavailable for selection.")

				# AI Refactoring for high complexity (whole snippet)
				if total_complexity > 10:
					st.header("🤖 AI Refactoring Suggestion")
					with st.spinner("Generating suggestion..."):
						suggestion = get_llm_refactoring_suggestion(cleaned, total_complexity)
					col_a, col_b = st.columns(2)
					with col_a:
						st.subheader("Original Code")
						st.code(cleaned, language="python")
					with col_b:
						st.subheader("Suggested Refactoring")
						st.code(suggestion, language="python")
