"""
 Created by Narayan Schuetz at 07/12/2018
 University of Bern

 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


SPECTRAL_MODEL_REGISTRY = {}


def Model(*args, **kwargs):
    """Decorator function, makes model definition a bit more obvious than relying on python's underscore variant"""

    if len(args) == 1 and callable(args[0]):
        cls = args[0]
        name = cls.__name__
        SPECTRAL_MODEL_REGISTRY[name] = cls
        return cls

    else:
        name = kwargs.get("name")

        if name is None:
            raise ValueError("Invalid argument, requires keyword argument 'name' if argument is given!")

        def wrapped_decorator(cls):
            SPECTRAL_MODEL_REGISTRY[name] = cls
            return cls

        return wrapped_decorator