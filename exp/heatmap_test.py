import numpy as np, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import somoclu, json
from mpl_toolkits.mplot3d import Axes3D


c1 = np.random.rand(50, 3)/5
c2 = (0.6, 0.1, 0.05) + np.random.rand(50, 3)/5
c3 = (0.4, 0.1, 0.7) + np.random.rand(50, 3)/5
data = np.float32(np.concatenate((c1, c2, c3)))
data = np.float32(np.concatenate((c1, c2, c3)))
colors = ["red"] * 50
colors.extend(["green"] * 50)
colors.extend(["blue"] * 50)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)
labels = range(150)
fig.savefig("/tmp/origin.png")


"""
greys = plt.get_cmap("Greys")


n_rows, n_columns = 70, 70
index = json.load(open('/tmp/event_analaysis_output/modeling/VectorSpaceModel_2016-12-31_2017-01-02', 'r'))
data = np.asarray(index["matrix"])
print data.shape
print data.T.shape
#som = somoclu.Somoclu(n_columns, n_rows, data=data.T, initialization="pca", maptype="planar", gridtype="rectangular")
#som = somoclu.Somoclu(n_columns, n_rows, data=data.T, initialization="pca", maptype="toroid", gridtype="hexagonal")
som = somoclu.Somoclu(n_columns, n_rows, data=data.T, initialization="pca", maptype="toroid", gridtype="hexagonal")
print "som train started"
som.train()
print "som train finished"
sys.stdout.flush()

#print "save view component"
#fig = som.view_component_planes(filename="/tmp/component.png", colormap=greys)
print "save umatrix view" 
fig = som.view_umatrix(bestmatches=True, filename="/tmp/umatrix.png", colormap=greys)

fig.close()
"""


