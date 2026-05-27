import atexit
import os
import pickle

import lmdb


_ENV_CACHE: dict[tuple[int, str], lmdb.Environment] = {}


def _cache_key(data_dir: str) -> tuple[int, str]:
    return os.getpid(), os.path.abspath(data_dir)


def get_lmdb_env(data_dir: str) -> lmdb.Environment:
    key = _cache_key(data_dir)
    env = _ENV_CACHE.get(key)
    if env is None:
        env = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        _ENV_CACHE[key] = env
    return env


def load_lmdb_split_keys(data_dir: str, mode: str):
    env = get_lmdb_env(data_dir)
    with env.begin(write=False) as txn:
        return list(pickle.loads(txn.get(b"__keys__"))[mode])


class LazyLMDBDatasetMixin:
    def _init_lmdb_dataset(self, data_dir: str, mode: str) -> None:
        self._data_dir = data_dir
        self._db_pid = None
        self.db = None
        self.keys = load_lmdb_split_keys(data_dir, mode)

    def _ensure_db_open(self) -> None:
        pid = os.getpid()
        if self.db is not None and self._db_pid == pid:
            return
        self.db = get_lmdb_env(self._data_dir)
        self._db_pid = pid


def _close_cached_envs() -> None:
    for env in _ENV_CACHE.values():
        try:
            env.close()
        except Exception:
            pass
    _ENV_CACHE.clear()


atexit.register(_close_cached_envs)
