import sys,numpy as np
sys.path.insert(0,"work")
from field_model import LocalField
q=np.array([[float(sys.argv[1]),float(sys.argv[2])]])
d,g=LocalField("work/final_ptv13.npz").evaluate(q)
print("A coordinates (px):",q[0])
print("Displacement x/y_down (px/pair):",d[0])
print("Source-coordinate displacement gradient [component,axis]:\n",g[0])
print("Raw estimate; inspect support/sensitivity masks before interpretation.")
