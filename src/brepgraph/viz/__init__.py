"""Optional Plotly visualization (``pip install brepgraph[viz]``).

Nothing here imports plotly at import time.
"""

from .graph_view import show_graph
from .show import show, show_face

__all__ = ["show", "show_face", "show_graph"]
