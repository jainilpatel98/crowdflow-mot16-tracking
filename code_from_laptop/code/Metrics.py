class Metrics:
    """Small base class used by the MATLAB MOT metrics wrapper.

    The original code you shared expects a parent class with a registry of
    metric names plus helpers to update values returned by the evaluator.
    This version keeps the API minimal and easy to understand.
    """

    def __init__(self):
        self._registry = {}

    def register(
        self,
        name,
        display_name=None,
        formatter=None,
        write_mail=True,
        write_db=True,
    ):
        """Register one metric and initialize its attribute on the instance."""
        self._registry[name] = {
            "display_name": display_name or name,
            "formatter": formatter,
            "write_mail": write_mail,
            "write_db": write_db,
        }
        setattr(self, name, 0)

    def update_values(self, update_dict):
        """Copy evaluator outputs into this object.

        The MATLAB devkit returns a dictionary-like structure with metric names
        and values. We mirror those values onto the Python object so the notebook
        can print or tabulate them later.
        """
        for key, value in dict(update_dict).items():
            setattr(self, key, value)

    def to_dict(self):
        """Return all registered metrics as a plain dictionary."""
        return {name: getattr(self, name, None) for name in self._registry}
