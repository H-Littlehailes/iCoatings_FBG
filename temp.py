# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
##os.chdir('C:/Users/hugh/PythonPrograms')
filename ='C:/Users/hugh/PythonPrograms/pi_million_digits.txt'
#filename = 'PythonPrograms/pi_digits.txt'
with open(filename) as file_object:
    lines = file_object.readlines()

#for line in lines:
 #   print(line.rstrip())
    
pi_string=''
for line in lines:
    pi_string += line.rstrip()
    
birthday = input("Enter your Birthday, in thr form mmddyyyy:   ")
if birthday in pi_string:
    print("Your birthday appears in the first million digits of pi")
else:
    print("Your birthday does not appear in the first million digits of pi")