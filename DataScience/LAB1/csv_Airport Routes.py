#!curl https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat \
#    -o airports.dat

#!curl https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat \
#    -o routes.dat

#!head -2 airports.dat

#!head -2 routes.dat

import csv
import matplotlib.pyplot as plt

airports = open("airports.dat", encoding='UTF-8')
routes = open("routes.dat", encoding='UTF-8')

airport_list = {}  # airport Id 와 airport name 매칭

for row in csv.reader(airports):
    airport_list[int(row[0])] = row[1]

Count = {}
Count_list = []

for row in csv.reader(routes):

    source = row[3]
    destination = row[5]

    if source == '\\N' or destination == '\\N':
        continue

    if int(source) not in Count_list:
        Count[int(source)] = 1
        Count_list.append(int(source))

    else:
        Count[int(source)] += 1

    if int(destination) not in Count_list:
        Count[int(destination)] = 1
        Count_list.append(int(destination))

    else:
        Count[int(destination)] += 1

Reverse = {x : y for y, x in Count.items()}

# find maximum label
Max = Count[2965]
for x, y in Count.items():
  if y >= Max:
    Max = y


# 가장 많이 방문된 Airport의 방문 횟수
print("Max Visited number of Airport :", Max)
# 해당 Airport의 ID
print("Max visited Airport ID : ", Reverse[Max])
# 해당 Airport의 이름
print("Max visited Airport Name :", airport_list[Max])

# Airport name 과 방문 횟수 matching
Result = {}

for (x, y), (w, z) in zip(Count.items(), airport_list.items()):

    if x in airport_list:
        visited_airport = airport_list[x]
        visited_num = y
        Result[visited_airport] = visited_num

# visualize

plt.figure(figsize=(30,10))
plt.barh(list(Result.keys()), list(Result.values()), color = 'g', height = 3)