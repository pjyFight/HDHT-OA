"""Simplified PretrainedConfig for unit testing."""


class PretrainedConfig:
    """Lightweight stand-in for the HF config base class."""

    model_type = ""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        """Return config attributes as dictionary."""
        return self.__dict__.copy()

