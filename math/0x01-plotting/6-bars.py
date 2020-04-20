#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

index = ('Farrah', 'Fred', 'Felicia')
width = 0.5

apples = fruit[0]
bananas = fruit[1]
oranges = fruit[2]
peaches = fruit[3]

plt.bar(index, apples, width, color='red', label='apples')
plt.bar(index, bananas, width, color='yellow', bottom=apples, label='bananas')
plt.bar(index, oranges, width, color='#ff8000', bottom=list(map(lambda x, y:
        x + y, apples, bananas)), label='oranges')
plt.bar(index, peaches, width, color='#ffe5b4', bottom=list(map(lambda x, y, z:
        x + y + z, apples, bananas, oranges)), label='peaches')
# another way: bottom=[x+y for x,y in zip(apples, bananas)]

plt.ylabel('Quantity of Fruit')
plt.yticks(np.arange(0, 80, step=10))
plt.title("Number of Fruit per Person")
plt.legend()
plt.show()
