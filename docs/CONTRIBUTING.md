# Contributing to VAM Tools

Thank you for your interest in contributing to VAM Tools! This guide will help you get started.

## Table of Contents

- [Development Setup](#development-setup)
- [Quality Gates](#quality-gates)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## Development Setup

### 1. Clone the Repository

```bash
git clone git@github.com:irjudson/vam-tools.git
cd vam-tools
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs:
- Production dependencies (click, rich, pillow, etc.)
- Development tools (pytest, black, isort, flake8, mypy)

### 4. Configure Git Hooks

Git hooks are automatically configured to run quality checks before pushing:

```bash
git config core.hooksPath .githooks
```

This is done automatically on clone, but verify it's set:

```bash
git config core.hooksPath
# Should output: .githooks
```

## Quality Gates

VAM Tools uses **automated quality gates** to ensure code quality. These gates run:

1. **Locally** - via pre-push hooks (catches issues before GitHub)
2. **In CI** - via GitHub Actions (final verification)

### Pre-Push Hook

Every `git push` runs these checks:

1. ✅ **Black** - Code formatting
2. ✅ **isort** - Import sorting
3. ✅ **flake8** - Linting
4. ✅ **pytest** - All 213 tests
5. ✅ **Common issues** - No debugger statements

**The pre-push hook matches CI exactly** - if it passes locally, CI will pass!

### GitHub Actions CI

The CI workflow runs on every push/PR:

**Quality Job** (fast-fail):
- Black formatting check
- isort import sorting
- flake8 linting
- mypy type checking (non-blocking)

**Test Job** (matrix):
- Python 3.9, 3.10, 3.11, 3.12
- Ubuntu and macOS
- Full test suite with coverage

## Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=vam_tools --cov-report=term
```

### Run Specific Test

```bash
pytest tests/shared/test_media_utils.py::TestFileTypeDetection::test_is_image_file
```

### Run Tests in Verbose Mode

```bash
pytest -v
```

### Run Tests Matching Pattern

```bash
pytest -k "checksum"
```

## Code Style

VAM Tools follows PEP 8 with some customizations.

### Formatting with Black

Black is our code formatter - it handles all formatting automatically:

```bash
# Check formatting
black --check vam_tools/ tests/

# Auto-format
black vam_tools/ tests/
```

**Configuration**: See `pyproject.toml` for Black settings.

### Import Sorting with isort

isort organizes imports:

```bash
# Check import sorting
isort --check-only vam_tools/ tests/

# Auto-sort imports
isort vam_tools/ tests/
```

**Configuration**: See `pyproject.toml` for isort settings (Black-compatible).

### Linting with flake8

flake8 catches code quality issues:

```bash
# Run linting
flake8 vam_tools/ tests/
```

**Configuration**: See `.flake8` for rules.

Common fixes:
- Line too long → Break into multiple lines
- Unused imports → Remove them
- Undefined names → Fix imports or typos

### Type Checking with mypy

mypy is optional but recommended:

```bash
mypy vam_tools/
```

**Note**: mypy failures don't block CI (yet), but we encourage type hints!

## Submitting Changes

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Or for bug fixes:

```bash
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- Write code
- Add/update tests
- Update documentation if needed

### 3. Run Quality Checks

```bash
# Format code
black vam_tools/ tests/
isort vam_tools/ tests/

# Check linting
flake8 vam_tools/ tests/

# Run tests
pytest
```

Or use the automated pre-commit check:

```bash
# This runs automatically on push, but you can test it:
.githooks/pre-push
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "Add feature: brief description

Longer description if needed.
- Bullet points for changes
- Why the change was made
"
```

**Commit Message Guidelines**:
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed description (if needed)
- Reference issues: "Fixes #123"

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

The **pre-push hook** will run automatically. If it passes, your code is ready for CI!

Create a Pull Request on GitHub with:
- Clear description of changes
- Why the change is needed
- How to test it
- Screenshots (if UI changes)

### 6. Address Review Feedback

If reviewers request changes:

```bash
# Make changes
git add .
git commit -m "Address review feedback"
git push
```

## Common Development Tasks

### Adding a New Utility Function

1. Add to `vam_tools/shared/media_utils.py` (if shared between V1/V2)
2. Export in `vam_tools/shared/__init__.py`
3. Add tests in `tests/shared/test_media_utils.py`
4. Run tests: `pytest tests/shared/`

### Adding a New CLI Command

1. Create in `vam_tools/cli/` (V1) or `vam_tools/v2/` (V2)
2. Import shared utilities from `vam_tools.shared`
3. Add integration tests in `tests/cli/`
4. Update README.md with usage examples

### Fixing a Bug

1. **Write a failing test first** (TDD)
2. Fix the bug
3. Ensure test passes
4. Add regression test if needed

### Updating Dependencies

```bash
# Update in pyproject.toml
# Then:
pip install -e ".[dev]"

# Test everything still works:
pytest
```

## Troubleshooting

### Pre-Push Hook Fails

**"Code formatting issues"**:
```bash
black vam_tools/ tests/
git add -u && git commit --amend --no-edit
```

**"Import sorting issues"**:
```bash
isort vam_tools/ tests/
git add -u && git commit --amend --no-edit
```

**"Linting issues"**:
```bash
# View issues
flake8 vam_tools/ tests/

# Fix manually, then:
git add -u && git commit --amend --no-edit
```

**"Tests failed"**:
```bash
# Run tests to see failures
pytest -v

# Fix tests, then:
git add -u && git commit -m "Fix tests"
```

### Skip Pre-Push Hook (Emergency Only)

```bash
git push --no-verify
```

⚠️ **Warning**: This may cause CI to fail! Only use in emergencies.

### CI Fails But Pre-Push Passed

This shouldn't happen (hooks match CI), but if it does:

1. Check GitHub Actions logs
2. Try specific Python version: `pyenv install 3.11 && pyenv local 3.11`
3. Run: `pytest -v`
4. File an issue if hooks and CI are out of sync

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/irjudson/vam-tools/issues)
- **Discussions**: [GitHub Discussions](https://github.com/irjudson/vam-tools/discussions)
- **Documentation**: See README.md and `/docs` folder

## Code of Conduct

Be respectful, inclusive, and professional. We're all here to learn and build great software together.

## License

By contributing, you agree that your contributions will be licensed under the project's license (see LICENSE file).

---

**Thank you for contributing to VAM Tools!** 🎉
