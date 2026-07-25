"""Vercel serverless entrypoint for the PronoiaQ web app.

Vercel's @vercel/python runtime serves an ASGI app exported as ``app``. The
FastAPI application lives in ``webapp/app/main.py`` and imports sibling code
trees (``QRC_code_stack``, ``Quantathon_stack``) by absolute path, so we add
``webapp`` to ``sys.path`` and hand Vercel the same app the local server runs.

Note: the interactive job-runner keeps run state in memory, which does not
persist across serverless invocations. On Vercel the read-only pages (home,
early-warning result, method, datasets, predictability) are the intended
surface; for the fully-live demo run locally or on a stateful host (see README).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBAPP = os.path.join(REPO_ROOT, "webapp")
if WEBAPP not in sys.path:
    sys.path.insert(0, WEBAPP)

from app.main import app  # noqa: E402  (path set above)

# Vercel looks for a module-level `app`.
__all__ = ["app"]
