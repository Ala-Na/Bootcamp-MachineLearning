from TinyStatistician import TinyStatistician
import numpy as np

# CAREFUL for errors in subjects examples (percentile, var, std)

ts = TinyStatistician()
a = np.array([1, 42, 300, 10, 59])
print(ts.mean(a))
print(np.mean(a), end="\n\n")

print(ts.median(a))
print(np.median(a), end="\n\n")


print(ts.quartile(a))
print(np.quantile(a, 0.25), np.quantile(a, 0.75), end="\n\n")


print(ts.percentile(a, 10))

print(ts.percentile(a, 15))

print(ts.percentile(a, 20))

print(ts.percentile(a, 28))

print(ts.percentile(a, 83), end="\n\n")

print(ts.var(a))
print(np.var(a), end="\n\n")

print(ts.std(a))
print(np.std(a), end="\n\n")

print(ts.mean(np.array([])))
