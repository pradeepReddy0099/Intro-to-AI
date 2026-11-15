# Your own datasets go here

Drop any `.csv`, `.tsv`, `.xlsx`, or `.xls` file in this folder and the
universal tests in `tests/test_any_dataset.py` will run the profiling,
cleaning, and full-pipeline checks on it — the same checks the built-in
tables go through, with no dataset-specific values assumed.

You can also point the tests at files elsewhere:

```bash
AGENT_TEST_DATA=/path/to/a/file.csv python -m pytest tests
AGENT_TEST_DATA=/path/to/a/folder python -m pytest tests
```
