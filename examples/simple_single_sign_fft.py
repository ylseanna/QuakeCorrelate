from os import environ
import numpy as np
from matplotlib.pyplot import figure, show

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

# template
vals = np.arange(0, 20, 0.1)
vals_template = vals[0:30]
vals_template
template = np.tan(vals_template)

# signal_1
signal_1 = np.zeros(len(vals))
signal_1[30:60] = template

# signal_2
signal_2 = np.roll(signal_1, -20)

# signal_3
signal_3 = np.roll(signal_1, 100) * -1

# result vectors
signal_vec = [signal_1, signal_2, signal_3]
rs = np.zeros(len(signal_vec))
lags = np.zeros(len(signal_vec))
corrgrams = np.zeros((len(signal_vec), len(signal_1)))

# call julia
jl.CCC.ncc(template, signal_vec, rs, lags, corrgrams, method="frequency")

# plot

fig = figure(figsize=(6,10))

#1
fr1 = fig.add_subplot(3, 1, 1)
fr1b = fr1.twinx()

fr1.plot(vals_template, template, color="black", linestyle="dashed")
fr1.plot(vals, signal_1, color="red")
fr1b.plot(vals, corrgrams[0], color="purple")

# 3
fr2 = fig.add_subplot(3, 1, 2)
fr2b = fr2.twinx()

fr2.plot(vals_template, template, color="black", linestyle="dashed")
fr2.plot(vals, signal_2, color="red")
fr2b.plot(vals, corrgrams[1], color="purple")

# 3
fr3 = fig.add_subplot(3, 1, 3)
fr3b = fr3.twinx()

fr3.plot(vals_template, template, color="black", linestyle="dashed")
fr3.plot(vals, signal_3, color="red")
fr3b.plot(vals, corrgrams[2], color="purple")

fig.savefig("simple_single_sign_fft.png")

# results
print(rs)
print(lags)
