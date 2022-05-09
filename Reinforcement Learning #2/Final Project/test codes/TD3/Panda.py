import numpy as np
import pandas as pd

data3 = np.array([0, -114.1 ,-110.1 ,-102.2 ,-84.3 ,-55.4 ,-38.1 ,-33.6 ,-21.8 ,-14.4 ,-59.3 ,-45.6 ,-35.6 ,25.3 ,25.5 ,250.3 ,281.5 ,244.2 ,246.4 ,289.4 ,248.0 ,294.1 ,295.4 ,296.1 ,251.7 ,301.2 ,299.7 ,298.0 ,296.0 ,254.4 ,293.0 ,296.2 ,295.5 ,266.9 ,294.9 ,281.2 ,173.8 ,-114.4 ,270.6 ,288.5 ,288.5 ,293.3 ,295.7 ,254.1 ,297.2 ,298.5 ,294.1 ,293.2 ,288.3 ,295.7 ])

data1 = data3[:10]
data2 = data3[10:20]
data3 = data3[20:30]
data4 = data3[30:40]
data5 = data3[40:]

dataset = [data1, data2, data3, data4, data5]
df = pd.DataFrame(dataset, index=[('0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100')], columns=list('0~100', '110~200', '210~300', '310~400', '410~500'))

print(df)