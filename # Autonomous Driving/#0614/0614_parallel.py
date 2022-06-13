import os
import sys
from multiprocessing import Process

def start_server(program):    
    os.system(program)

programs = ["python3 0614_1_control.py", "./a.out"]

if __name__ == '__main__':
    for program in programs:
        proc = Process(target  = start_server, args = (program,))        
        proc.start()


