"""
models/__init__.py
------------------
Registry of all trajectory prediction models.

Usage
-----
    from models import get_model

    model = get_model("d_pool",    weights_path="path/to/weights.state")
    model = get_model("social_lstm", weights_path="path/to/weights.pth")
    model = get_model("autobot",   weights_path="path/to/weights.pth")
    model = get_model("eq_motion", weights_path="path/to/weights.pth")
    model = get_model("transformer")   # no pretrained weights needed for training
"""

from models.d_pool      import DPoolModel
from models.social_lstm import SocialLSTMModel
from models.autobot     import AutobotModel
from models.eq_motion   import EqMotionModel
from models.transformer import TransformerModel


_REGISTRY = {
    "d_pool":       DPoolModel,
    "social_lstm":  SocialLSTMModel,
    "autobot":      AutobotModel,
    "eq_motion":    EqMotionModel,
    "transformer":  TransformerModel,
}


def get_model(model_name: str, **kwargs):
    """Instantiate a model by name.

    Parameters
    ----------
    model_name : str  one of 'd_pool', 'social_lstm', 'autobot',
                      'eq_motion', 'transformer'
    **kwargs   : passed directly to the model constructor
                 (e.g. weights_path, obs_length, pred_length, device)

    Returns
    -------
    model instance with a `.predict(scene, goal) -> Tensor` method
    """
    if model_name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[model_name](**kwargs)


__all__ = [
    "get_model",
    "DPoolModel",
    "SocialLSTMModel",
    "AutobotModel",
    "EqMotionModel",
    "TransformerModel",
]
