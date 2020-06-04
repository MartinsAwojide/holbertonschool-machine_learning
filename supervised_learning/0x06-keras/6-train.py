#!/usr/bin/python3
import sys

input = {'200': 0,
         '301': 0,
         '400': 0,
         '401': 0,
         '403': 0,
         '404': 0,
         '405': 0,
         '500': 0}
status_code = ""
file_size = ""
counter = 0
for line in sys.stdin:
    aux = line.split()
    status_code = aux[-2]
    file_size += int(aux[-1])
    if status_code in input:
        input[status_code] += 1
    if counter % 10 == 0:
        print(file_size)
    counter += 1
    print(input)