#!/usr/bin/env python3
"""
Bundle Codebase Script

Runs during build time to bundle all relevant source files into
a single JSON file for AI diagnosis context.

Usage:
    python scripts/bundle_codebase.py
"""

import os
import re
import json
import glob
from datetime import datetime
from pathlib import Path


# Files to bundle
FILES_TO_BUNDLE = [
    "agent_sql_final_1212_patched.py",
    "app_server.py",
    "frontend/src/App.js",
    "services/*.py",
]

# Additional patterns to search for Python files
ADDITIONAL_PATTERNS = [
    "*.py",  # Root Python files
]

# Files to exclude
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pyc",
    "node_modules",
    ".git",
    "build",
    ".env",
    "venv",
    "test_*.py",
    "*_test.py",
]

# Output file
OUTPUT_FILE = "codebase_context.json"

# Max file size (500KB)
MAX_FILE_SIZE = 500 * 1024


def should_exclude(file_path: str) -> bool:
    """Check if a file should be excluded."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in file_path:
            return True
    return False


def extract_tools(content: str) -> list:
    """Extract tool names from agent code."""
    tools = []

    # Pattern for @tool decorator functions
    tool_pattern = r'@tool\s*(?:\([^)]*\))?\s*\ndef\s+(\w+)'
    matches = re.findall(tool_pattern, content)
    tools.extend(matches)

    # Pattern for Tool() objects
    tool_obj_pattern = r'Tool\s*\(\s*name\s*=\s*["\']([^"\']+)["\']'
    matches = re.findall(tool_obj_pattern, content)
    tools.extend(matches)

    # Pattern for SQL_AGENT_TOOLS list
    tools_list_pattern = r'SQL_AGENT_TOOLS\s*=\s*\[([\s\S]*?)\]'
    match = re.search(tools_list_pattern, content)
    if match:
        # Extract function names from the list
        func_pattern = r'\b(\w+_tool|\w+_query|\w+_data)\b'
        funcs = re.findall(func_pattern, match.group(1))
        tools.extend(funcs)

    return list(set(tools))


def extract_endpoints(content: str) -> list:
    """Extract API endpoints from FastAPI code."""
    endpoints = []

    # Pattern for FastAPI route decorators
    patterns = [
        r'@app\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
        r'@router\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content)
        for method, path in matches:
            endpoints.append(f"{method.upper()} {path}")

    return list(set(endpoints))


def read_file_safe(file_path: str) -> str:
    """Read a file safely, handling encoding issues."""
    try:
        # Check file size
        if os.path.getsize(file_path) > MAX_FILE_SIZE:
            print(f"  Skipping large file: {file_path}")
            return ""

        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
            return ""
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return ""


def find_files(base_dir: str, patterns: list) -> list:
    """Find files matching patterns."""
    files = set()

    for pattern in patterns:
        full_pattern = os.path.join(base_dir, pattern)
        matches = glob.glob(full_pattern, recursive=True)
        for match in matches:
            if os.path.isfile(match) and not should_exclude(match):
                # Convert to relative path
                rel_path = os.path.relpath(match, base_dir)
                files.add(rel_path)

    return sorted(files)


def main():
    """Bundle codebase into JSON context file."""
    print("=" * 60)
    print("Bundling codebase for AI diagnosis context")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    print(f"Base directory: {base_dir}")

    # Find all files to bundle
    all_patterns = FILES_TO_BUNDLE + ADDITIONAL_PATTERNS
    files_to_bundle = find_files(".", all_patterns)

    print(f"\nFound {len(files_to_bundle)} files to bundle:")
    for f in files_to_bundle:
        print(f"  - {f}")

    # Bundle files
    bundled_files = {}
    all_tools = []
    all_endpoints = []

    for file_path in files_to_bundle:
        content = read_file_safe(file_path)
        if content:
            bundled_files[file_path] = content

            # Extract metadata
            if file_path.endswith('.py'):
                tools = extract_tools(content)
                endpoints = extract_endpoints(content)
                all_tools.extend(tools)
                all_endpoints.extend(endpoints)

                if tools:
                    print(f"  Found {len(tools)} tools in {file_path}")
                if endpoints:
                    print(f"  Found {len(endpoints)} endpoints in {file_path}")

    # Create context object
    context = {
        "bundled_at": datetime.now().isoformat(),
        "files": bundled_files,
        "tools": list(set(all_tools)),
        "endpoints": list(set(all_endpoints)),
        "file_count": len(bundled_files),
    }

    # Write output
    output_path = os.path.join(base_dir, OUTPUT_FILE)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=2)

    # Calculate size
    size_kb = os.path.getsize(output_path) / 1024

    print(f"\n{'=' * 60}")
    print(f"Bundle complete!")
    print(f"  Output: {output_path}")
    print(f"  Files bundled: {len(bundled_files)}")
    print(f"  Tools extracted: {len(context['tools'])}")
    print(f"  Endpoints extracted: {len(context['endpoints'])}")
    print(f"  Bundle size: {size_kb:.1f} KB")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
