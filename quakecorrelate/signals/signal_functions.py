from juliacall import Main as jl


def random_event(*args, **kwargs):
    return jl.RandomEvent(*args, **kwargs)

def add_padding(*args, **kwargs):
    return jl.AddPadding(*args, **kwargs)

def add_noise(*args, **kwargs):
    return jl.AddNoise(*args, **kwargs)