"""13-DOF articulator definitions for the OpenJaw environment.

Following Anand et al. (2025) "Teaching Machines to Speak Using Articulatory Control":
  - Tongue dorsum (X, Y): Genioglossus posterior, styloglossus
  - Tongue blade  (X, Y): Genioglossus middle, verticalis
  - Tongue tip    (X, Y): Genioglossus anterior, superior longitudinal
  - Lower incisor (X, Y): Masseter, pterygoids, digastric (jaw proxy)
  - Upper lip     (X, Y): Orbicularis oris superior, levator labii
  - Lower lip     (X, Y): Orbicularis oris inferior, depressor labii
  - Vocal loudness (1D):  Cricothyroid, thyroarytenoid (glottal amplitude)
"""

from openjaw.core.mdp import ARTICULATOR_NAMES, ARTICULATORS, Articulator

# Re-export for convenience
__all__ = ["ARTICULATORS", "ARTICULATOR_NAMES", "Articulator", "get_articulator_groups"]


def get_articulator_groups() -> dict[str, list[int]]:
    """Get articulator indices grouped by body part.

    Returns:
        Dictionary mapping body part names to lists of DOF indices.
    """
    return {
        "tongue_dorsum": [0, 1],
        "tongue_blade": [2, 3],
        "tongue_tip": [4, 5],
        "jaw": [6, 7],
        "upper_lip": [8, 9],
        "lower_lip": [10, 11],
        "voicing": [12],
    }
