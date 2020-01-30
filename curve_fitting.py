import numpy as np
def curve(points):
	x = points[:,0]
	y = points[:,1]

	z = np.polyfit(x, y, 2)
	f = np.poly1d(z)

	x_new = np.linspace(x[0], x[-1], 100)
	y_new = f(x_new)
	return list(zip(x_new, y_new))
