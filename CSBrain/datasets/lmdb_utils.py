import atexit
import os
from typing import Dict, Tuple

import lmdb


_ENV_CACHE: Dict[Tuple[int, str], lmdb.Environment] = {}


def get_lmdb_env(path: str) -> lmdb.Environment:
    cache_key = (os.getpid(), path)
    env = _ENV_CACHE.get(cache_key)
    if env is None:
        env = lmdb.open(
            path,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )
        _ENV_CACHE[cache_key] = env
    return env


def close_all_lmdb_envs() -> None:
    for env in _ENV_CACHE.values():
        try:
            env.close()
        except Exception:
            pass
    _ENV_CACHE.clear()


atexit.register(close_all_lmdb_envs)
