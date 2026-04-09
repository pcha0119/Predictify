# Rule: Build Validation

**After every code change, validate the build before considering the task done.**

## What to run

```bash
# 1. Syntax check every changed .py file
python -m py_compile <changed_file.py>

# 2. Import check if any backend/ file changed
cd Sales_Forecasting_ver1 && python -c "from backend.api.app import app; print('import ok')"

# 3. If tests exist for the changed module, run them
```

If any step fails, fix it before moving on. Do not present work to the user while the build is red.

## Updating this rule

When a new validation step is discovered to be necessary (e.g. a linting check, a migration, a smoke test), add it here. Ask permission before adding, consistent with the project's CLAUDE.md update policy.
