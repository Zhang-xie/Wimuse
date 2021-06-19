from .ARIL import ARIL
from .CSIDA import CSIDA
from .Widar import Widar

__factory = {
    'widar':Widar,
    'aril':ARIL,
    'csida':CSIDA,
}


def names():
    return sorted(__factory.keys())


def get_full_name(name):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name].__name__


def create(name, *args, **kwargs):
    # root = args['root']
    # if root is not None:
    #     root = root/ get_full_name(name)
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](*args, **kwargs)
