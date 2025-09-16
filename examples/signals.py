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
    
    newmodule(environ.get("CCC_PATH"))


jl.seval("using CCC")

jl.seval("using CCC.Signals")

samp_freq = 500
event_duration = 1

# call julia
event, etime = jl.CCC.RandomEvent(np.arange(1,5), np.arange(1, 180), np.arange(0.1, 1, 0.1), samp_freq, event_duration, 3)

# plot

fig = figure(figsize=(6,10))

#1
fr1 = fig.add_subplot(3, 1, 1)
fr1.plot(etime, np.real(event), color="black")

# # 3
# fr2 = fig.add_subplot(3, 1, 2)

# fr2.plot(vals_template, template, color="black")

# # 3
# fr3 = fig.add_subplot(3, 1, 3)
# fr3.plot(vals_template, template, color="black")

fig.savefig("signals.png")

