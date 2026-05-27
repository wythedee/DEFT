import io
import pickle


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def loads_lmdb_pickle(data):
    try:
        return pickle.loads(data)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("numpy._core"):
            return _NumpyCompatUnpickler(io.BytesIO(data)).load()
        raise
