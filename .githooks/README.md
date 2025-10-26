# Git Hooks

This directory contains git hooks to ensure code quality before pushing.

## Setup

The hooks are automatically configured when you clone the repository. If needed, run:

```bash
git config core.hooksPath .githooks
```

## Pre-Push Hook

The `pre-push` hook runs before every `git push` and performs:

1. **Black formatting check** - Ensures code is formatted
2. **Pytest** - Runs all tests
3. **Common issues check** - Checks for debugger statements, print statements in core code

### If Pre-Push Hook Fails

**Formatting issues:**
```bash
black vam_tools/ tests/
git add -u
git commit --amend --no-edit
git push
```

**Test failures:**
Fix the failing tests, then:
```bash
git add -u
git commit -m "Fix tests"
git push
```

**Debugger statements found:**
Remove `import pdb`, `pdb.set_trace()`, or `breakpoint()` calls:
```bash
# Find them
grep -r "import pdb\|breakpoint()" vam_tools/

# Remove them, then
git add -u
git commit --amend --no-edit
git push
```

## Skipping Hooks (Not Recommended)

In emergencies only:
```bash
git push --no-verify
```

**⚠️ Warning:** Skipping hooks may cause CI failures on GitHub!

## Benefits

- ✅ Catches issues before they reach GitHub
- ✅ CI workflows always succeed
- ✅ Faster feedback loop (local vs waiting for CI)
- ✅ Prevents broken code in main branch
