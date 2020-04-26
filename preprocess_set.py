import os
import csv


with open('./data/label.csv') as f :
    csv_reader = csv.reader(f)
    lines = [x for x in csv_reader]

print(lines[:5])

new_label = []
for line in lines:
    one = []
    one.append(line[0])
    if line[1] == 'ang':
        one.append(line[1])
        one.append('0')
    elif line[1] == 'hap':
        one.append(line[1])
        one.append('1')
    elif line[1] == 'exc':
        one.append('hap')
        one.append('1')
    elif line[1] == 'sad':
        one.append(line[1])
        one.append('2')
    elif line[1] == 'neu':
        one.append(line[1])
        one.append('3')
    else:
        continue
    new_label.append(one)

print(len(new_label))
with open('./data/label_process.csv', 'w') as f:
    for n in new_label:
        f.write('{},{},{}\n'.format(n[0],n[1],n[2]))


