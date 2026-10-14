"""رخش · Rakhsh — a reflex-based, single-forward retrieval & decision core.

>>> from rakhsh import Registry
>>> reg = Registry("~/.rakhsh")
>>> brain = reg.brain("books")
>>> brain.ingest("shahnameh.txt", open("shahnameh.txt", "rb").read()); brain.wait()
>>> brain.ask("رستم با چه کسی جنگید؟")["answer"]["text"]
"""
from ._version import __version__
from .brain import Brain
from .config import BrainConfig, DeciderConfig, FeaturizerConfig, ModelConfig, TrustPolicy
from .decide import Decider
from .errors import NotFound, NotReady, RakhshError
from .ingest import Chunk, ingest, register_loader
from .registry import Registry

__all__ = ["__version__", "Brain", "Decider", "Registry", "BrainConfig", "DeciderConfig", "FeaturizerConfig",
           "ModelConfig", "TrustPolicy", "RakhshError", "NotFound", "NotReady", "Chunk", "ingest", "register_loader"]
