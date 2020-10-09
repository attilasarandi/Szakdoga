import pandas as pd
import numpy as np

df = pd.DataFrame({'A':[1,2,3],
                   'B':[4,5,6],
                   'C':[7,8,9],
                   'D':[1,3,5],
                   'E':[5,3,6],
                   'F':[7,4,3]})

'print(df.iloc[::-1])'

input = 0.2
a = np.append(input, -input)
print(a)