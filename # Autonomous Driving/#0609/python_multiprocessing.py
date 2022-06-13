import os
import sys
from multiprocessing import Process

def start_server(program):    
    os.system(program)

programs = ["python3 test.py", "python3 test2.py"]
#programs = ["./a.out", "python3 0609_complete_parallel.py"]

if __name__ == '__main__':
    for program in programs:
        proc = Process(target  = start_server, args = (program,))        
        proc.start()



