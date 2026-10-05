"""What dew imports to find sparx's registrations: the `dew.plugins` entry point.

Dew imports a plugin's entry module when a name its own index lacks is
looked up (a run's model, objective, dataset or a shared kind's member), and
reads only what that import registers. This module imports every sparx
module that registers something.
"""

from sparx import datasets, dew, models, nn, surrogate  # noqa: F401 - imported to register

__all__: list[str] = []
