from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(n, variance, step=2, correlation=False):
    val = 1
    ys =[]
    for i in range(n):
        y = val +random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys,dtype=np.float64)

def best_fit_line(xs,yx):
    m = (mean(xs)*mean(ys) - mean(xs*ys)) / (mean(xs)*mean(xs) - mean(xs*xs))
    b = mean(ys) - m*mean(xs)
    return m,b

def squared_error(ys, regression_line):
    return sum((regression_line - ys)**2)

## calculates coefficient of determination(r squared) closer it is to number 1 the more accurate/precise
def coefficient_of_determination(ys, regression_line):
    y_mean_line = [mean(ys) for y in ys]
    squared_error_regr = squared_error(ys, regression_line)
    squared_errr_y_mean = squared_error(ys, y_mean_line)
    return 1 - (squared_error_regr / squared_errr_y_mean)
xs, ys = create_dataset(40, 10, 2, correlation = 'pos')
m,b = best_fit_line(xs,ys)
regression_line = [m*x + b for x in xs]

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)
plt.scatter(xs,ys)
plt.plot(xs, regression_line)
plt.show()
