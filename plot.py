import matplotlib.pyplot as plt
import numpy as np
import re

file1 = 'testmse.log'
file2 = 'testrmse.log'
legend = ['MSE','RMSE']
xlabel = 'Epoch'
title = 'Error for FCN Dense Prediction'

with open(file1) as f:
    content = f.readlines()

def integer(str):
	num = re.search('([0-9].[0-9]*)e\\'+ str[7]+'([0-9]*)',str)
	return float(num.group(1)) * 10**(int(str[7]+'1')*float(num.group(2)))

content = [x.split() for x in content[3:]] 
# print content
plotter = []
for j in content:
	plotter.append([integer(i) for i in j])

plotter = np.array(plotter)

with open(file2) as f2:
    content2 = f2.readlines()

content2 = [x.split() for x in content2[3:]] 
# print content
plotter2 = []
for j in content2:
	plotter2.append([integer(i) for i in j])

plotter2 = np.array(plotter2)

plt.plot(plotter,'.-',plotter2,'*-')
plt.legend(legend)
plt.xlabel(xlabel)
plt.title(title) 
plt.show()


