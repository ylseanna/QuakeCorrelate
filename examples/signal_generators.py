import numpy as np
from matplotlib.pyplot import figure

import quakecorrelate

samp_freq = 500
event_duration = 1

# call julia
event, etime = quakecorrelate.signals.random_event(
    np.arange(1, 5),
    np.arange(1, 180),
    np.arange(0.1, 1, 0.1),
    samp_freq,
    event_duration,
    3,
)

trace, time = quakecorrelate.signals.add_padding(event, 3, 2)

event_noise = quakecorrelate.signals.add_noise(
    trace,
    np.arange(20, 35, 0.1),
    np.arange(1, 180),
    np.arange(0.1, 0.2, 0.01),
    20,
)


# plot

fig = figure(figsize=(12, 12))

# 1
fr1 = fig.add_subplot(3, 1, 1)
fr1.margins(x=0)
fr1.plot(etime, np.real(event), color="black")

# 2
fr2 = fig.add_subplot(3, 1, 2)
fr2.margins(x=0)

fr2.plot(time, np.real(trace), color="black")

# 3
fr3 = fig.add_subplot(3, 1, 3)
fr3.margins(x=0)

fr3.plot(time, np.real(event_noise), color="black")

fig.savefig("signals.png", bbox_inches='tight')
