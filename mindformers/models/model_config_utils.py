"""Helpers for registering MindFormers model configuration parameters."""

from functools import wraps

from mindformers.parallel_core.mf_model_config import MFModelConfig


def register_mf_model_parameter(*, mf_model_kwargs: MFModelConfig):
    """Decorator attaching MF model metadata to the init method."""

    def decorator(function):
        function._mf_model_kwargs = mf_model_kwargs  # pylint: disable=protected-access
        return function

    return decorator


def ignore_and_delete_parameter():
    """Decorator pass-through used for compatibility."""

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    return decorator

