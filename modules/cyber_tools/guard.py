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


def guard(target, authorized, action, store=None, profile=None):
    """Raise PermissionError unless the active action is allowed on target.

    Allowed without ``authorized=True`` when the target is loopback/private
    or when ``store.covered(profile, target)`` confirms a valid consent
    profile (non-expired, target listed).
    """
    if target is None:
        raise ValueError(f"{action}: a target is required")
    if authorized:
        return
    if _targets_local(target):
        return
    if store is not None and profile is not None \
            and store.covered(profile, target):
        return
    raise PermissionError(
        f"{action}: refusing to touch {target!r} \u2014 only loopback/private "
        "hosts are scanned by default. Pass authorized=True to confirm you "
        "own or have permission to test this target, or --profile <name> "
        "with a previously granted consent profile.")


def require_authorization(action):
    def deco(fn):
        def wrapper(target, *args, **kwargs):
            guard(target, kwargs.get("authorized", False), action)
            return fn(target, *args, **kwargs)
        return wrapper
    return deco
