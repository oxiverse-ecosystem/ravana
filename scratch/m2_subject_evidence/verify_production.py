"""Re-run the EXACT pre-registered gate script against the production module."""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["PYTHONPATH"] = ("C:/Users/Likhith/Documents/projects/"
                            "ravana_m2_subject_evidence/ravana/src;"
                            + os.environ.get("PYTHONPATH", ""))
import experiment as ex  # noqa: E402
from ravana.learn import subject_type_evidence as prod  # noqa: E402

ex.ste = prod
ex.RESULTS = str(HERE / "results_production.json")
ex.main()
