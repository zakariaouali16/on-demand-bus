"""On-demand bus dispatcher (demand-responsive transit)."""
from .watsonx import _load_local_config

# Read .env (DRT_CITY, WATSONX_*) and apikey.json once, for every entry point.
_load_local_config()
