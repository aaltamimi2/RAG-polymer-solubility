# DISSOLVE npm CLI Distribution Spec

Status: planning spec only

Target repository: `RAG-polymer-solubility`

Target Python package: `strap-agent`

Target CLI entry point: `dissolve = "strap.agent:main"`

## Purpose

Provide a safe npm-based entry point so new users can install and launch the DISSOLVE CLI without manually cloning the repository or editing their Python environment.

The npm package should act as a launcher and bootstrapper for the Python CLI. It should not become the scientific runtime itself.

## Recommended User Experience

Primary install flow:

```bash
npm install -g @dissolve-ai/cli
dissolve setup
dissolve doctor
dissolve
```

One-shot install flow:

```bash
npx @dissolve-ai/cli setup
dissolve
```

Existing Python-native install flow should remain supported:

```bash
pipx install strap-agent
dissolve
```

## Safety Requirements

The installer must be safe by default.

Required safety properties:

- No npm `postinstall` script that downloads or installs Python packages automatically.
- No `sudo`.
- No mutation of system Python, global site-packages, conda base, or user shell startup files.
- No `curl | bash` style installation.
- No opaque shell command strings when a structured spawn API can be used.
- All Python dependencies install into an isolated managed virtual environment.
- The managed virtual environment lives under a user-owned app/cache directory.
- Downloads must use HTTPS package indexes or versioned GitHub release artifacts.
- Production releases should install pinned package versions.
- Optional heavy dependencies should be explicit, not silently installed.
- Diagnostic output should show exactly where DISSOLVE was installed and which Python executable is used.

Unsafe patterns to avoid:

```json
{
  "scripts": {
    "postinstall": "pip install git+https://github.com/..."
  }
}
```

```bash
sudo npm install -g ...
pip install --user ...
python setup.py install
```

## Distribution Architecture

The npm package provides a Node.js wrapper with a `dissolve` bin command.

The wrapper is responsible for:

- Locating a compatible Python interpreter.
- Creating a managed virtual environment.
- Installing the published Python package into that environment.
- Forwarding `dissolve` CLI invocations to the Python entry point.
- Running diagnostics through `dissolve doctor`.
- Updating or removing the managed installation on explicit user command.

The Python package remains responsible for:

- Agent runtime.
- Subagent orchestration.
- Optimization, separation, TEA/LCA, safety, and visualization tools.
- Domain dependencies and scientific computation.

## Proposed Folder Layout

Future implementation can live under `npm/` or a separate package workspace:

```text
npm/
  package.json
  README.md
  bin/
    dissolve.js
  lib/
    paths.js
    python.js
    bootstrap.js
    doctor.js
    spawn-dissolve.js
  test/
    bootstrap.test.js
    path-normalization.test.js
```

This `npm-setup/` folder is intentionally only the planning/spec folder.

## npm Package Metadata

Candidate package names:

- `@dissolve-ai/cli`
- `@strap-agent/dissolve`
- `dissolve-cli`

Recommended first choice:

```json
{
  "name": "@dissolve-ai/cli",
  "version": "0.1.0",
  "description": "Safe npm launcher for the DISSOLVE scientific agent CLI",
  "type": "module",
  "bin": {
    "dissolve": "bin/dissolve.js"
  },
  "engines": {
    "node": ">=18"
  },
  "files": [
    "bin",
    "lib",
    "README.md",
    "LICENSE"
  ],
  "scripts": {
    "test": "node --test",
    "lint": "node --check bin/dissolve.js"
  }
}
```

Do not add `postinstall`.

## Managed Install Location

Use an OS-appropriate app data/cache directory.

Linux:

```text
${XDG_DATA_HOME:-~/.local/share}/dissolve/venvs/default
```

macOS:

```text
~/Library/Application Support/dissolve/venvs/default
```

Windows:

```text
%LOCALAPPDATA%\DISSOLVE\venvs\default
```

WSL should resolve to the Linux path inside the distribution.

The wrapper should print the resolved path during `dissolve setup` and `dissolve doctor`.

## Commands

`dissolve`

- If setup is complete, forwards arguments to the Python CLI.
- If setup is missing, prints a short message explaining `dissolve setup`.
- Should not perform a silent install.

`dissolve setup`

- Creates or refreshes the managed virtual environment.
- Installs the pinned Python package.
- Verifies that the Python `dissolve` entry point works.

`dissolve doctor`

- Reports Node version.
- Reports Python executable.
- Reports managed venv location.
- Reports installed `strap-agent` version.
- Reports whether `dissolve` resolves to the managed CLI.
- Reports optional feature availability where cheap to check.
- Reports configured environment variables without printing secret values.

`dissolve update`

- Upgrades the managed Python package to the npm package's expected Python package version.
- Does not upgrade across incompatible major versions without explicit confirmation.

`dissolve uninstall-runtime`

- Removes the managed virtual environment.
- Does not uninstall the npm package itself.

`dissolve env`

- Prints paths and environment details useful for debugging.

## Python Package Source Policy

Preferred production source:

```text
PyPI package: strap-agent==<version>
```

Acceptable pre-publication source:

```text
GitHub release wheel URL pinned by version and SHA256
```

Avoid for public release:

```text
git+https://github.com/aaltamimi2/RAG-polymer-solubility.git@branch
```

Rationale: branch installs are not reproducible and are harder to audit.

## Bootstrap Algorithm

High-level `dissolve setup` flow:

1. Resolve app data path.
2. Locate Python:
   - Prefer `python3`.
   - On Windows, try `py -3`.
   - Require Python `>=3.10`.
3. Create venv:
   - `python -m venv <managed_venv_path>`.
4. Upgrade safe packaging tools inside venv:
   - `<venv_python> -m pip install --upgrade pip`.
5. Install DISSOLVE:
   - Production: `<venv_python> -m pip install strap-agent==<expected_version>`.
   - Pre-release: install versioned wheel artifact with hash check.
6. Verify:
   - `<venv_python> -m pip show strap-agent`.
   - `<venv_dissolve> --version` if a version flag exists.
   - Otherwise run a lightweight import check.
7. Write metadata:
   - `runtime.json` with npm version, Python package version, venv path, install timestamp, and source.

## Version Coupling

The npm package and Python package should declare compatible versions.

Example:

```json
{
  "dissolve": {
    "pythonPackage": "strap-agent",
    "pythonVersion": "0.2.0"
  }
}
```

The wrapper should refuse to run if the managed Python package is absent or incompatible, and instruct the user to run:

```bash
dissolve setup
```

## Optional Dependencies

The Python package currently defines optional extras:

- `viz`
- `ml`
- `biosteam`
- `rag`
- `literature`
- `all`

Recommended npm setup modes:

```bash
dissolve setup
dissolve setup --extras biosteam,ml
dissolve setup --full
```

Default should be conservative. Do not silently install large ML or chemistry extras unless they are required for the intended public CLI baseline.

Before publication, decide whether the npm default maps to:

```text
strap-agent
```

or:

```text
strap-agent[biosteam,ml]
```

## Secret Handling

The npm wrapper must never ask users to paste secrets into command history by default.

Recommended behavior:

- Support `.env` files through the Python app as today.
- `dissolve doctor` should detect whether keys are configured but redact values.
- Output examples should use placeholders.

Example diagnostic output:

```text
LANGSMITH_API_KEY: set, redacted
ANTHROPIC_API_KEY: missing
GOOGLE_API_KEY: missing
```

## Cross-Platform Requirements

Initial supported environments:

- Linux
- WSL2 Ubuntu
- macOS

Windows native should be supported after explicit testing. The wrapper should fail clearly if a platform is not supported.

Path requirements:

- Accept normal POSIX paths.
- Accept Windows paths where Node runs on Windows.
- Do not translate WSL UNC paths in the npm wrapper unless Node is running on Windows and explicitly needs translation.
- Leave domain path normalization to the Python runtime where possible.

## Observability

Setup should write a metadata file:

```json
{
  "schema_version": "1.0",
  "npm_package": "@dissolve-ai/cli",
  "npm_version": "0.1.0",
  "python_package": "strap-agent",
  "python_version": "0.2.0",
  "venv_path": "...",
  "python_executable": "...",
  "installed_at": "...",
  "source": "pypi"
}
```

Do not upload telemetry from the npm installer.

LangSmith tracing remains a Python runtime feature controlled by environment configuration.

## Testing Plan

Unit tests:

- Python discovery succeeds/fails predictably.
- Venv path resolution is platform-correct.
- Wrapper refuses to run when setup is missing.
- Wrapper forwards arguments after setup.
- `doctor` redacts secrets.
- No `postinstall` script exists.

Integration tests:

- Fresh Linux install.
- Fresh WSL install.
- Re-run setup idempotently.
- Update from older runtime metadata.
- Uninstall managed runtime.
- Run `dissolve --help` or equivalent smoke command.

Security tests:

- Verify no shell startup files are edited.
- Verify no global Python packages are installed.
- Verify no command uses `sudo`.
- Verify secrets are redacted.

Publication checks:

- `npm pack --dry-run`.
- `npm publish --dry-run`.
- Python wheel builds with `python -m build`.
- Python wheel installs into a clean venv.
- Checksums are recorded for release artifacts if using GitHub wheels.

## Implementation Milestones

Milestone 1: Python package readiness

- Ensure `strap-agent` builds a wheel.
- Add `dissolve --version` if not already available.
- Confirm clean install into a fresh venv.
- Decide baseline extras for public users.

Milestone 2: npm wrapper scaffold

- Add npm package structure.
- Implement `dissolve`, `setup`, `doctor`, and runtime metadata.
- Keep all install actions explicit.

Milestone 3: Linux/WSL validation

- Test on the group-meeting environment without relying on editable installs.
- Test from a clean temporary user cache path.
- Confirm no global Python mutation.

Milestone 4: pre-publication release

- Publish a beta npm package or use `npm pack`.
- Install from the packed tarball.
- Run DISSOLVE smoke queries.

Milestone 5: production publication

- Publish Python package or release wheel.
- Publish npm package.
- Add user-facing install docs.

## Open Decisions Before Publication

- Final npm package name.
- Final Python package name exposed to public users.
- Whether to publish `strap-agent` as-is or rename/alias to a DISSOLVE-branded package.
- Whether default npm setup installs only core dependencies or includes BioSTEAM and ML extras.
- Whether native Windows is supported in the first public release or WSL-only on Windows.
- Whether installation source is PyPI or GitHub release wheels for the first release.

## Recommended First Implementation

Do not implement a Codex-style full binary distribution first.

Recommended first implementation:

```text
npm wrapper + explicit setup + isolated Python venv + pinned Python wheel
```

This is the best balance of user convenience, device safety, and implementation speed.
