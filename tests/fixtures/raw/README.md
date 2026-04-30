# Raw response fixtures

JSON files under this tree are **golden examples** for offline extractor tests (`tests/gateway/test_extractor_offline.py`).

- Prefer shapes taken from **real API responses** (capture with `pytest --live --provider <id> --pretty-print -s`, see **Testing** in the repo `README.md`) or from the vendor’s own API docs.
- Trim secrets and oversized fields before committing.

When an upstream response shape changes, update the matching JSON here and adjust `src/gateway/extractor.py` in the same change.
