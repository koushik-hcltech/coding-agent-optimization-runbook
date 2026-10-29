# Runbook: Hill-Climbing a Coding Agent (Evaluation-Driven Harness Optimization)

## 1. Overview
This runbook consolidates the end-to-end workflow for diagnosing and optimizing a minimalistic Google ADK coding agent (`agent.py`) using a fixed, lightweight model (`gemini-3.1-flash-lite` with `ThinkingLevel.LOW`).

Because the model and thinking budget are fixed, all reliability gains come from optimizing the **agent harness**:
1. **Tools** (workspace discovery, codebase search, surgical editing, and sandboxed command execution).
2. **System Instructions** (requiring full codebase exploration, root-cause tracing, edge-case handling, and execution-based self-verification).
3. **Sampling Configuration** (lowering `temperature` from `1.0` to `0.2` for deterministic tool use).

---

## 2. Step-by-Step Workflow Performed

### Task 1: Environment Setup & Baseline Agent Sanity Check
```bash
# Activate virtual environment
source .venv/bin/activate

# Smoke-test the coding agent on a simple prompt
.venv/bin/python agent.py "Write a hello world python script"
```

### Task 2: Run the Baseline Evaluation
```bash
# Run all 5 evaluation tasks in parallel
.venv/bin/python eval.py

# Or run a single task during iterative debugging
.venv/bin/python eval.py --task invoice_report
```
Each evaluation run creates `runs/<timestamp>/<task_id>/` containing:
- `run.json`: Pass/fail status, tool call count, duration, and stop reason.
- `tool_uses.json`: Flattened sequence of tool calls and responses for quick inspection.
- `trajectory.evalset.json`: Full ADK trajectory.
- `src/`: The final code shipped by the agent.
- `repro.sh`: Script that runs the hidden test (`tests/hidden.py`) against the shipped `src/`.

### Task 3: Manual Trajectory Analysis (`invoice_report`)
1. **Reproduce the failure**:
   ```bash
   bash runs/<timestamp>/invoice_report/repro.sh
   ```
   Output showed `AssertionError: 124.0` when `125.49` was expected.
2. **Inspect `tool_uses.json`**:
   - The baseline agent only read `src/invoices/report.py` (the file named in the prompt), wrote a `reproduce_issue.py` script that it could only `read_file` back (no execution tool), and rewrote `report.py` twice using `Decimal`.
   - It never discovered `src/invoices/load.py`, where `parse_amount` truncated cents via `int(float(raw))`.
3. **Submit Diagnosis**:
   ```text
   /submit-task-diagnosis For invoice_report, I observed ...; the coding agent failed because ...
   ```

### Task 4: Automated Trajectory Analysis Across Remaining Failures
Inspecting the remaining failed tasks in `runs/<timestamp>/` revealed the same core harness gaps:
- **`reminder_time` (Bug Fix)**: The agent read `src/scheduling/core.py` and wrote 7 unrunnable Python scripts (`list_files.py`, `search.py`, etc.) trying to list files in `src/scheduling/`, never finding `src/scheduling/parsing.py` where `.replace(tzinfo=None)` stripped timezone offsets.
- **`range_list` (Build CLI)**: The agent wrote `src/expand.py` in a single `write_file` call using `range(start, end + 1)` and never ran it, missing descending ranges like `9-7 -> 9 8 7`.
- **`interval_set` & `expense_report` (Build CLI)**: Require strict `Decimal` half-up rounding (`ROUND_HALF_UP`), trailing-zero trimming, empty-stdin handling (`0.00`), and exact error exit codes (`exit 1` / `exit 2` with `stderr`).

### Task 5: Harness Optimization (`agent.py`) & Validation
We upgraded `agent.py` with four additional sandboxed tools and a structured `SYSTEM_INSTRUCTION`, then re-ran `.venv/bin/python eval.py` (achieving 4/5 tasks passed) and graded the agent via `/grade-my-agent` (scoring **100/100**).

---

## 3. Reusable Code & Harness Patterns (`agent.py`)

### A. Sandboxed Path Resolution
Always resolve paths relative to the sandbox root and block `../` traversal:
```python
def _resolve(sandbox: str, path: str) -> str:
    sandbox = os.path.realpath(sandbox)
    full = os.path.realpath(os.path.join(sandbox, path))
    if full != sandbox and not full.startswith(sandbox + os.sep):
        raise ValueError(f"path escapes sandbox: {path!r}")
    return full
```

### B. Essential Coding Agent Tools
Return tool errors as strings inside `{"result": ...}` rather than raising exceptions so the LLM can see errors and self-correct:

1. **`list_files(directory: str = ".")`**: Recursively walks the sandbox (ignoring `__pycache__`, `.git`, `.venv`, `.pyc`) so the agent can discover the full repository layout immediately.
2. **`search_files(pattern: str, directory: str = ".")`**: Searches for regex/substring matches across files and returns `path:lineno: line` so the agent can trace imports and callers upstream.
3. **`read_file(path: str)` & `write_file(path: str, contents: str)`**: Reads and writes text files relative to the sandbox.
4. **`edit_file(path: str, old_text: str, new_text: str)`**: Replaces exact text blocks without rewriting entire files.
5. **`run_command(command: str)`**: Executes shell/Python commands inside the sandbox with a timeout and returns `exit_code`, `stdout`, and `stderr`:
   ```python
   def _run_command(sandbox: str, command: str) -> str:
       try:
           proc = subprocess.run(
               command,
               shell=True,
               cwd=os.path.realpath(sandbox),
               capture_output=True,
               text=True,
               timeout=15,
               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.path.realpath(sandbox)},
           )
           return (
               f"exit_code: {proc.returncode}\n"
               f"stdout:\n{proc.stdout}\n"
               f"stderr:\n{proc.stderr}"
           )
       except subprocess.TimeoutExpired:
           return "Error: command timed out after 15 seconds"
       except Exception as exc:
           return f"Error: {exc}"
   ```

### C. System Prompt Structure (`SYSTEM_INSTRUCTION`)
A high-performing coding agent system prompt enforces three phases:
1. **Explore & Understand**: Call `list_files(".")` first, read sibling modules and `__init__.py`, and use `search_files` to trace upstream parsing/loading helpers instead of trusting that the bug is in the file named in the prompt.
2. **Implement Carefully**: Always write files via tools (never chat-only code blocks), use `decimal.Decimal` with `ROUND_HALF_UP` for currency/decimal formatting, preserve timezone offsets, handle both ascending and descending ranges, and obey exact CLI `stderr` + exit code contracts.
3. **Verify & Test Edge Cases**: Run code via `run_command` against prompt examples AND edge cases (empty stdin, single-element, descending ranges, touching intervals, malformed inputs) and iterate until all checks pass, cleaning up temporary test scripts afterward.
