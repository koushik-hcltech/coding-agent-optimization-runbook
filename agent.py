"""Coding agent with tools for reading, writing, editing, searching files, and running commands."""

import os
import re
import subprocess
import sys
import warnings

from google.adk.agents import Agent
from google.genai import types

# The default model, shared with eval.py so there is a single source of truth.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# Silencing an ADK warning to declutter terminal output
warnings.filterwarnings("ignore", message=r".*JSON_SCHEMA_FOR_FUNC_DECL.*", category=UserWarning)

def _resolve(sandbox: str, path: str) -> str:
    """Turn a sandbox-relative path into an absolute one, refusing to escape the sandbox."""
    sandbox = os.path.realpath(sandbox)
    full = os.path.realpath(os.path.join(sandbox, path))
    # Allow only the sandbox root itself or a path beneath it (blocks "../" escapes).
    if full != sandbox and not full.startswith(sandbox + os.sep):
        raise ValueError(f"path escapes sandbox: {path!r}")
    return full


# The helpers below return errors as plain strings instead of raising, so the
# model sees what went wrong and can recover on its next turn.

def _read(sandbox: str, path: str) -> str:
    try:
        with open(_resolve(sandbox, path), encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return f"Error: no such file: {path}"
    except (ValueError, OSError) as exc:
        return f"Error: {exc}"


def _write(sandbox: str, path: str, contents: str) -> str:
    try:
        full = _resolve(sandbox, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(contents)
    except (ValueError, OSError) as exc:
        return f"Error: {exc}"
    return f"Wrote {len(contents)} chars to {path}"


def _edit(sandbox: str, path: str, old_text: str, new_text: str) -> str:
    try:
        full = _resolve(sandbox, path)
        with open(full, encoding="utf-8") as fh:
            content = fh.read()
        if old_text not in content:
            return f"Error: old_text not found in {path}"
        updated = content.replace(old_text, new_text, 1)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(updated)
        return f"Edited {path} successfully"
    except FileNotFoundError:
        return f"Error: no such file: {path}"
    except (ValueError, OSError) as exc:
        return f"Error: {exc}"


def _list_files(sandbox: str, directory: str = ".") -> str:
    try:
        root = _resolve(sandbox, directory or ".")
        if not os.path.exists(root):
            return f"Error: no such directory: {directory}"
        if os.path.isfile(root):
            return os.path.relpath(root, os.path.realpath(sandbox))
        entries = []
        sandbox_real = os.path.realpath(sandbox)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in {"__pycache__", ".git", ".venv"})
            for fname in sorted(filenames):
                if fname.endswith(".pyc"):
                    continue
                full = os.path.join(dirpath, fname)
                entries.append(os.path.relpath(full, sandbox_real))
        return "\n".join(entries) if entries else "(empty directory)"
    except (ValueError, OSError) as exc:
        return f"Error: {exc}"


def _search_files(sandbox: str, pattern: str, directory: str = ".") -> str:
    try:
        root = _resolve(sandbox, directory or ".")
        if not os.path.exists(root):
            return f"Error: no such directory: {directory}"
        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))
        matches = []
        sandbox_real = os.path.realpath(sandbox)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in {"__pycache__", ".git", ".venv"})
            for fname in sorted(filenames):
                if fname.endswith(".pyc"):
                    continue
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, sandbox_real)
                try:
                    with open(full, encoding="utf-8") as fh:
                        for lineno, line in enumerate(fh, 1):
                            if regex.search(line):
                                matches.append(f"{rel}:{lineno}: {line.rstrip()}")
                except (OSError, UnicodeDecodeError):
                    continue
        return "\n".join(matches) if matches else "No matches found."
    except (ValueError, OSError) as exc:
        return f"Error: {exc}"


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


SYSTEM_INSTRUCTION = """You are an autonomous software engineering agent working inside a project sandbox.
Follow a disciplined workflow on every task:

1. EXPLORE & UNDERSTAND:
   - First call `list_files(".")` to see all files in the workspace.
   - Read every relevant file in `src/` (including `__init__.py` and sibling modules), not just the file mentioned in the prompt.
   - Use `search_files` to find references, imports, and upstream/downstream functions across the codebase. Bugs often originate in upstream parsing, loading, or helper modules rather than the function where the symptom is observed.

2. IMPLEMENT CAREFULLY:
   - Always create or modify files using `write_file` or `edit_file`. Never only output code in chat.
   - Follow every requirement in the prompt strictly:
     * Exact file paths and function signatures.
     * Exact numeric semantics (e.g., use `decimal.Decimal` with `ROUND_HALF_UP` when exact decimal arithmetic or half-up rounding is required; avoid `int(...)` truncation when parsing decimal amounts; preserve timezone offsets when parsing datetimes).
     * Exact CLI behavior: stdin parsing, ignoring blank lines, sorting/tie-breaking rules, trimming trailing zeros vs fixed decimal places, writing error messages to `sys.stderr`, and exiting with the exact required exit code.
     * For ranges or intervals, handle both ascending (`a <= b`) and descending (`a > b`, stepping down from `a` to `b` inclusive) when expanding `a to b inclusive`, unless the spec explicitly says `start > end` is an error.

3. VERIFY & TEST EDGE CASES:
   - Use `run_command` (with `python3 ...`) to execute your code and verify behavior before finishing.
   - For bug fixes, test the full pipeline end-to-end (including upstream parsing/loading functions and downstream functions) to confirm the root cause is fixed.
   - For CLI programs, test:
     * All examples from the prompt.
     * Empty input (e.g., empty stdin / zero records, ensuring commands like `total`, `average`, or `length` print `0.00` without crashing).
     * Single-element, equal-bound, touching, negative, and descending/reversed inputs.
     * Malformed input lines and invalid/missing CLI commands (verify non-zero exit code and stderr output).
   - If any test fails, inspect the error, fix the code, and re-run until all checks pass.
   - Remove any temporary test files you created outside `src/` before finishing."""


def build_agent(sandbox: str, model: str) -> Agent:
    # Each tool's docstring is the description the model sees, so it states that
    # paths are sandbox-relative. All tools return their payload under "result".
    def list_files(directory: str = ".") -> dict:
        """List all files recursively under a sandbox-relative directory (default '.')."""
        return {"result": _list_files(sandbox, directory)}

    def search_files(pattern: str, directory: str = ".") -> dict:
        """Search for a regex or substring pattern across files under a sandbox-relative directory."""
        return {"result": _search_files(sandbox, pattern, directory)}

    def read_file(path: str) -> dict:
        """Read a text file (path is relative to the project root) and return its contents."""
        return {"result": _read(sandbox, path)}

    def write_file(path: str, contents: str) -> dict:
        """Create or overwrite a text file (path is relative to the project root)."""
        return {"result": _write(sandbox, path, contents)}

    def edit_file(path: str, old_text: str, new_text: str) -> dict:
        """Replace the first occurrence of old_text with new_text in a sandbox-relative file."""
        return {"result": _edit(sandbox, path, old_text, new_text)}

    def run_command(command: str) -> dict:
        """Run a shell command (e.g. `python3 ...`) inside the project sandbox and return exit_code, stdout, and stderr."""
        return {"result": _run_command(sandbox, command)}

    return Agent(
        name="coding_agent",
        model=model,
        instruction=SYSTEM_INSTRUCTION,
        tools=[list_files, search_files, read_file, write_file, edit_file, run_command],
        generate_content_config=types.GenerateContentConfig(
            temperature=0.2,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW
            )
        )
    )

if __name__ == "__main__":
    # Test the agent on a prompt in ./out sandbox.  Usage: python agent.py "your prompt"
    import asyncio
    import subprocess
    import sys
    from google.adk.runners import InMemoryRunner

    # Default to Vertex AI (like eval.py) so `python agent.py` works without a
    # Gemini API key; an existing env value, if set, takes precedence.
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "qwiklabs-gcp-04-e45d13a60ec0")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

    sandbox = os.path.join(os.path.dirname(os.path.realpath(__file__)), "out")
    os.makedirs(os.path.join(sandbox, "src"), exist_ok=True)
    prompt = " ".join(sys.argv[1:]) or "Write a hello world python script"

    async def _run_agent() -> None:
        runner = InMemoryRunner(agent=build_agent(sandbox, DEFAULT_MODEL))
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="me")
        message = types.Content(role="user", parts=[types.Part(text=prompt)])
        async for event in runner.run_async(user_id="me", session_id=session.id, new_message=message):
            for part in (event.content and event.content.parts) or []:
                print(part.text or "", end="", flush=True)
    asyncio.run(_run_agent())
    import glob
    for script in glob.glob(f"{sandbox}/**/*.py", recursive=True):
        print(f"\n=== output of {os.path.relpath(script, sandbox)} ===\n" + subprocess.run([sys.executable, script], cwd=sandbox, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout, end="")
