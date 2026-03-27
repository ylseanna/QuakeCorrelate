import numpy as np
from matplotlib.pyplot import figure

import quakecorrelate

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

# run cross correlation
results, lags, corrgrams = quakecorrelate.ncc(template, signal_vec, method="frequency", return_corrgrams=True)

# plot

fig = figure(figsize=(6,6))

#1
fr1 = fig.add_subplot(3, 1, 1)
fr1.margins(x=0)
fr1b = fr1.twinx()

fr1.plot(vals_template, template, color="black", linestyle="dashed")
fr1.plot(vals, signal_1, color="red")
fr1b.plot(vals, corrgrams[0], color="purple")

# 3
fr2 = fig.add_subplot(3, 1, 2)
fr2.margins(x=0)
fr2b = fr2.twinx()

fr2.plot(vals_template, template, color="black", linestyle="dashed")
fr2.plot(vals, signal_2, color="red")
fr2b.plot(vals, corrgrams[1], color="purple")

# 3
fr3 = fig.add_subplot(3, 1, 3)
fr3.margins(x=0)
fr3b = fr3.twinx()

fr3.plot(vals_template, template, color="black", linestyle="dashed")
fr3.plot(vals, signal_3, color="red")
fr3b.plot(vals, corrgrams[2], color="purple")

fig.savefig("simple_single_sign_fft.png", bbox_inches='tight')

# results
print(rs)
print(lags)
