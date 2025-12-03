"""Utility classes for Glm4Moe models."""


class Glm4MoePreTrainedModel:
    """Lightweight base class for Glm4Moe model hierarchy."""

    config_class = None

    def __init__(self, config=None):
        self.config = config

