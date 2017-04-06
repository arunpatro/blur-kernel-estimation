import matplotlib.pyplot as plt
import numpy as np
import re


with open('./test.log') as f:
    content = f.readlines()

def integer(str):
	num = re.search('([0-9].[0-9]*)e\-([0-9]*)',str)
	return float(num.group(1)) * 10**(-1*float(num.group(2)))

content = [x.split() for x in content[1:]] 
# print content
plotter = []
for j in content:
	plotter.append([integer(i) for i in j])

plotter = np.array(plotter)
'''
with open('./benchmark-cuda.log') as f2:
    content2 = f2.readlines()

content2 = [x.split() for x in content2[1:]] 
# print content
plotter2 = []
for j in content2:
	plotter2.append([integer(i) for i in j])

plotter2 = np.array(plotter2)
'''
plt.plot(plotter)
plt.show()



