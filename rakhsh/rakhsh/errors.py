class RakhshError(Exception):
    """Base error. Messages are user-facing (Persian + English where helpful)."""


class NotReady(RakhshError):
    pass


class NotFound(RakhshError):
    pass
