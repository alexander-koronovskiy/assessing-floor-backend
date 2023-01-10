import dataclasses
import json

from numpy import ndarray


class ResourceEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ndarray):
            return o.tolist()

        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        return super().default(o)
