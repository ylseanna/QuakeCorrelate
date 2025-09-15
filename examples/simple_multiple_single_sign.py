from os import environ
import numpy as np
import matplotlib.pyplot as plt

from juliacall import Main as jl
from juliacall import newmodule
from juliacall import Pkg as jlPkg

from dotenv import load_dotenv

load_dotenv()

# set julia environments
if environ.get("CCC_TYPE") == "path":
    # activate environment
    jlPkg.activate(environ.get("CCC_PATH"))

    # activate package
    newmodule(environ.get("CCC_PATH"))


jl.seval("using CCC")


vals = np.arange(1, 20, 0.1)


vals_template = vals[20:50]

vals_template

template = np.tan(vals_template)

plt.plot(template)

# signal_1

signal_1 = np.zeros(len(vals))

signal_1[50:80] = template

plt.plot(signal_1)

# signal_2

signal_2 = np.zeros(len(vals))

signal_2[120:150] = -template

plt.plot(signal_2)


signal_vec = [signal_1, signal_2]
rs = np.zeros(len(signal_vec))
lags = np.zeros(len(signal_vec))

jl.CCC.ncc(template, signal_vec, rs, lags)

plt.save_fig ()
print(rs)
print(lags)
