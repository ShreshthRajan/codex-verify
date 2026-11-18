"""
Claude API Client for Multi-Agent Code Verification

Provides async Claude Sonnet 4.5 API access with:
- Specialized prompts per agent type
- Rate limiting and retry logic
- Cost tracking
- Error handling
- JSON structured output parsing
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class LLMResponse:
    """Parsed LLM response"""
    issues: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    tokens_used: int
    cost_usd: float
    latency_seconds: float


class ClaudeClient:
    """
    Async Claude Sonnet 4.5 client for code analysis

    Features:
    - Async/await for parallel agent execution
    - Rate limiting (50 req/min)
    - Exponential backoff retry (3 attempts)
    - Cost tracking
    - Specialized prompts per agent type
    """

    # Model selection
    MODEL = "claude-sonnet-4-5-20250929"

    # Pricing (per million tokens)
    INPUT_COST_PER_MTOK = 3.00
    OUTPUT_COST_PER_MTOK = 15.00

    # Rate limits
    MAX_REQUESTS_PER_MINUTE = 50

    # Agent-specific prompts
    PROMPTS = {
        'correctness': """You are an expert Python code correctness analyst with 15+ years experience finding subtle bugs.

Analyze this code for:
1. Logic errors (off-by-one, wrong operators, incorrect conditions)
2. Edge case handling (None, empty lists/strings, negative numbers, zero division)
3. Exception handling (missing try/except around risky operations: file I/O, network, parsing)
4. Input validation (missing checks on function parameters)
5. Resource leaks (unclosed files, sockets, database connections)
6. Contract violations (docstring promises vs actual implementation)

For EACH issue found, return:
{{
  "type": "logic_error|edge_case|exception_handling|input_validation|resource_leak|contract_violation",
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "line": <line_number or null>,
  "message": "Clear description of the bug and its impact",
  "confidence": 0.0-1.0
}}

Code to analyze:
```python
{code}
```

Return ONLY valid JSON array: [{{"type": "...", "severity": "...", ...}}]
""",

        'security': """You are an expert security auditor specializing in OWASP Top 10 and CWE vulnerability detection.

Analyze this Python code for security vulnerabilities. Check for:

INJECTION FLAWS:
- CWE-89: SQL injection (string formatting in queries, f-strings with user input)
- CWE-78: OS command injection (os.system, subprocess with user input)
- CWE-94: Code injection (eval, exec, compile with user input)
- CWE-611: XML external entities (xml.etree.parse, lxml without defusedxml)
- CWE-643: XPath injection

AUTHENTICATION & ACCESS:
- CWE-306: Missing authentication (public endpoints without auth checks)
- CWE-798: Hardcoded credentials (passwords, API keys, tokens in source)
- CWE-327: Weak cryptography (MD5, SHA1, DES)

FILE & RESOURCE:
- CWE-22: Path traversal (../ in paths, os.path.join without sanitization)
- CWE-434: Unrestricted file upload (no extension/MIME validation)
- CWE-502: Unsafe deserialization (pickle.loads, yaml.load without SafeLoader)

NETWORK & DATA:
- CWE-918: Server-side request forgery (requests.get with user-controlled URL)
- CWE-400: Resource exhaustion (unbounded loops, no limits)

For EACH vulnerability, return:
{{
  "type": "sql_injection|command_injection|code_execution|xml_vuln|path_traversal|hardcoded_secret|weak_crypto|ssrf|unsafe_deserialization|file_upload|missing_auth|dos",
  "cwe_id": "CWE-XXX",
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "line": <line_number or null>,
  "message": "Vulnerability description and exploit scenario",
  "confidence": 0.0-1.0
}}

Code to analyze:
```python
{code}
```

Return ONLY valid JSON array: [{{"type": "...", "cwe_id": "CWE-XXX", ...}}]
""",

        'performance': """You are an algorithmic complexity and performance optimization expert.

Analyze this code for:
1. Time complexity (identify O(n), O(n²), O(2ⁿ) patterns)
2. Space complexity (memory usage, large data structures)
3. Inefficient algorithms:
   - Nested loops that could be optimized
   - Repeated computations (missing memoization)
   - N+1 query problems
   - String concatenation in loops
4. Resource exhaustion (CWE-400):
   - Unbounded loops
   - No pagination limits
   - Infinite recursion risks

For EACH issue, return:
{{
  "type": "complexity|inefficient_algorithm|resource_exhaustion|memory_leak",
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "line": <line_number or null>,
  "message": "Performance issue and optimization suggestion",
  "complexity": "O(...)" if applicable,
  "confidence": 0.0-1.0
}}

Code to analyze:
```python
{code}
```

Return ONLY valid JSON array: [{{"type": "...", "severity": "...", ...}}]
""",

        'style': """You are a Python code quality expert focusing on maintainability and best practices.

Analyze this code for:
1. PEP 8 violations (naming, line length, import organization)
2. Missing docstrings (functions, classes, modules)
3. Poor naming (unclear variable/function names)
4. Code smells (duplicated code, long functions, deep nesting)
5. Documentation gaps

For EACH issue, return:
{{
  "type": "pep8|naming|documentation|code_smell",
  "severity": "LOW|MEDIUM",
  "line": <line_number or null>,
  "message": "Quality issue and improvement suggestion",
  "confidence": 0.0-1.0
}}

Note: Style issues should be LOW or MEDIUM severity only (never block deployment).

Code to analyze:
```python
{code}
```

Return ONLY valid JSON array: [{{"type": "...", "severity": "...", ...}}]
"""
    }

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude client

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.client = AsyncAnthropic(api_key=self.api_key)

        # Rate limiting
        self.request_times = []
        self.total_cost_usd = 0.0
        self.total_tokens = 0

    async def _rate_limit(self):
        """Ensure we don't exceed 50 req/min"""
        now = time.time()

        # Remove requests older than 60 seconds
        self.request_times = [t for t in self.request_times if now - t < 60]

        # If at limit, wait
        if len(self.request_times) >= self.MAX_REQUESTS_PER_MINUTE:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.request_times.append(now)

    async def analyze_code(
        self,
        code: str,
        agent_type: str,
        temperature: float = 0.7,
        max_retries: int = 3
    ) -> LLMResponse:
        """Analyze code using Claude with specialized prompt for agent type

        Args:
            code: Python code to analyze
            agent_type: 'correctness'|'security'|'performance'|'style'
            temperature: Sampling temperature (0.0-1.0)
            max_retries: Number of retry attempts on failure

        Returns:
            LLMResponse with parsed issues
        """
        if agent_type not in self.PROMPTS:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Get specialized prompt
        prompt = self.PROMPTS[agent_type].format(code=code)

        # Retry logic
        for attempt in range(max_retries):
            try:
                # Rate limit
                await self._rate_limit()

                # Call Claude API
                start_time = time.time()

                response = await self.client.messages.create(
                    model=self.MODEL,
                    max_tokens=4000,
                    temperature=temperature,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )

                latency = time.time() - start_time

                # Extract response
                response_text = response.content[0].text

                # Parse JSON
                try:
                    # Handle markdown code blocks if present
                    if '```json' in response_text:
                        response_text = response_text.split('```json')[1].split('```')[0]
                    elif '```' in response_text:
                        response_text = response_text.split('```')[1].split('```')[0]

                    response_text = response_text.strip()
                    issues = json.loads(response_text)

                    if not isinstance(issues, list):
                        issues = []

                except json.JSONDecodeError:
                    # Fallback: return empty if can't parse
                    issues = []

                # Calculate cost
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                cost = (input_tokens * self.INPUT_COST_PER_MTOK / 1_000_000 +
                       output_tokens * self.OUTPUT_COST_PER_MTOK / 1_000_000)

                self.total_cost_usd += cost
                self.total_tokens += (input_tokens + output_tokens)

                return LLMResponse(
                    issues=issues,
                    metadata={
                        'agent_type': agent_type,
                        'temperature': temperature,
                        'model': self.MODEL
                    },
                    tokens_used=input_tokens + output_tokens,
                    cost_usd=cost,
                    latency_seconds=latency
                )

            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed, return empty
                    return LLMResponse(
                        issues=[],
                        metadata={'error': str(e), 'agent_type': agent_type},
                        tokens_used=0,
                        cost_usd=0.0,
                        latency_seconds=0.0
                    )

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            'total_requests': len(self.request_times),
            'total_cost_usd': self.total_cost_usd,
            'total_tokens': self.total_tokens,
            'avg_cost_per_request': self.total_cost_usd / len(self.request_times) if self.request_times else 0
        }
