import yaml


class DataProvider(dict):
    pass


class ConfigProvider(DataProvider):
    def __contains__(self, k):
        try:
            return dict.__contains__(self, k) or hasattr(self, k)
        except:
            return False

    # only called if k not found in normal places
    def __getattr__(self, k):
        try:
            # Throws exception if not in prototype chain
            return object.__getattribute__(self, k)
        except AttributeError:
            try:
                return self[k]
            except KeyError as exc:
                raise AttributeError(k) from exc

    def __setattr__(self, k, v):
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                self[k] = v
            except Exception as exc:
                raise AttributeError(k) from exc
        else:
            object.__setattr__(self, k, v)

    def __delattr__(self, k):
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                del self[k]
            except KeyError as exc:
                raise AttributeError(k) from exc
        else:
            object.__delattr__(self, k)

    @staticmethod
    def from_dict(d):
        def structure(x):
            if isinstance(x, dict):
                return ConfigProvider((k, structure(v)) for k, v in dict.items(x))
            elif isinstance(x, (list, tuple)):
                return type(x)(structure(v) for v in x)
            else:
                return x
        return structure(d)

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path) as config_file:
            config = yaml.unsafe_load(config_file)
        return cls.from_dict(config)
