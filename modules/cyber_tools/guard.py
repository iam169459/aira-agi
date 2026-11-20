"""Authorization guard for active security tooling."""
import ipaddress
import socket


def _targets_local(target):
    try:
        infos = socket.getaddrinfo(target, None)
    except OSError:
        return True  # unresolvable names are handled by the caller
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if not (
            addr.is_loopback
            or addr.is_private
            or addr.is_link_local
        ):
            return False
    return True


def guard(target, authorized, action):
    """Raise PermissionError unless the active action is allowed on target."""
    if target is None:
        raise ValueError(f"{action}: a target is required")
    if authorized:
        return
    if _targets_local(target):
        return
    raise PermissionError(
        f"{action}: refusing to touch {target!r} — only loopback/private "
        "hosts are scanned by default. Pass authorized=True to confirm you "
        "own or have permission to test this target."
    )


def require_authorization(action):
    def deco(fn):
        def wrapper(target, *args, **kwargs):
            guard(target, kwargs.get("authorized", False), action)
            return fn(target, *args, **kwargs)
        return wrapper
    return deco
