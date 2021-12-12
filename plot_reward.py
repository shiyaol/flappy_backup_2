import matplotlib.pyplot as plt

x = list()
y = list()

file1 = open('reward_output_newnn.txt','r')

lines = file1.readlines()
for line in lines:
    y.append(int(line.split(',')[2]))
    x.append(int(line.split(',')[0]))

plt.plot(x, y)
plt.show()