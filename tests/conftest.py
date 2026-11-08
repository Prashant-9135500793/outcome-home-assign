"""Test isolation: point the service at a throwaway DB and upload dir before the app is imported."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="extract-tests-")
os.environ["DB_PATH"] = os.path.join(_tmp, "jobs.db")
os.environ["UPLOAD_DIR"] = os.path.join(_tmp, "uploads")
os.environ.setdefault("LLM_API_KEY", "test-key")

import pytest

SAMPLE_DOCS = os.path.join(os.path.dirname(__file__), "..", "sample", "sample-data-documents")
# The sample documents are provided data and are not committed; tests that read them skip without them.
requires_samples = pytest.mark.skipif(not os.path.isdir(SAMPLE_DOCS), reason="sample documents not present")
