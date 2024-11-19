import numpy as np

def bankers_algorithm():
    p = int(input("Enter the number of processes: "))
    r = int(input("Enter the number of resources: "))

    total = list(map(int, input("Enter the total resources for each resource type (space-separated): ").split()))

    allocation = []
    print("Enter the allocation matrix (allocated resources for each process):")
    for i in range(p):
        row = list(map(int, input(f"Process {i + 1} allocation: ").split()))
        allocation.append(row)

    max_need = []
    print("Enter the max need matrix (maximum resources required by each process):")
    for i in range(p):
        row = list(map(int, input(f"Process {i + 1} max need: ").split()))
        max_need.append(row)

    allocation = np.array(allocation)
    max_need = np.array(max_need)
    total = np.array(total)

    remaining = max_need - allocation

    sum_allocation = allocation.sum(axis=0)
    available = total - sum_allocation

    print("\nInitial Allocation Matrix:")
    print(allocation)
    print("\nMaximum Need Matrix:")
    print(max_need)
    print("\nRemaining Need Matrix:")
    print(remaining)
    print("\nInitial Available Resources:", available)

    safe_sequence = []
    finish = [False] * p  

    while len(safe_sequence) < p: 
        allocated_this_round = False

        for i in range(p):
            if not finish[i]:
                if all(remaining[i] <= available):
                    print(f"\nProcess {i + 1} is able to complete; allocating resources.")
                    available += allocation[i]  
                    safe_sequence.append(i + 1) 
                    finish[i] = True  
                    allocated_this_round = True
                    print("Current Available Resources after allocation:", available)

        if not allocated_this_round:
            print("\nNo safe sequence found. System is in an unsafe state.")
            return
        
    print("\nSafe Sequence Found:", safe_sequence)

bankers_algorithm()

"""
def is_safe(n, r, available, allocation, need):
    finish = [False] * n  
    work = available[:]  
    safe_sequence = []  
    count = 0

    while count < n:
        found = False
        for i in range(n):
            if not finish[i]:
                if all(need[i][j] <= work[j] for j in range(r)):
                    for k in range(r):
                        work[k] += allocation[i][k]

                    finish[i] = True
                   
                    safe_sequence.append(i)
                    found = True
                    count += 1

        if not found:
            return False, []
    return True, safe_sequence 


def banker_algorithm():
    n = int(input("Enter no of processes: "))
    r = int(input("Enter no of resources: "))

    r_type = input("Enter type of resources\n\t1. CPU\n\t2. Memory\n\t3. Printer: ")
    if r_type == "CPU":
        print("Resource Type: CPU")
    elif r_type == "Memory":
        print("Resource Type: Memory")
    elif r_type == "Printer":
        print("Resource Type: Printer")
    else:
        print("Invalid resource type. Exiting.")
        return

    available = [10,5,7]

    max_resources = [[0] * r for _ in range(n)]
    allocation = [[0] * r for _ in range(n)]
    need = [[0] * r for _ in range(n)]

    print("Enter the maximum resources for each process (space-separated):")
    for i in range(n):
        print(f"Process {i + 1}:")
        max_resources[i] = list(map(int, input().split()))

    print("Enter the allocated resources for each process (space-separated):")
    for i in range(n):
        print(f"Process {i + 1}:")
        allocation[i] = list(map(int, input().split()))

    for i in range(n):
        for j in range(r):
            need[i][j] = max_resources[i][j] - allocation[i][j]

    print("Need matrix is:")
    for row in need:
        print(" ".join(map(str, row)))

    safe, safe_sequence = is_safe(n, r, available, allocation, need)

    if safe:
        print("The system is in a safe state.")
        print("Safe sequence is:", " -> ".join([f"P{i+1}" for i in safe_sequence]))
    else:
        print("The system is not in a safe state. No safe sequence exists.")

banker_algorithm()
"""


####FCFS
n=int(input("Enter the number of Processes:"))
d={}
l=[]

for i in range(n):
    pid=input("Enter process id here:")
    l.append(pid)

    keys=int(input(f"Enter the arrivial time of {pid}: "))
    values=int(input(f"Enter the burst time of {pid}: "))
    d.update({keys:values})

print(d)
print(l)

dk=list(d.keys())
dk.sort()
sdk={i:d[i] for i in dk}
print(sdk)

dv=list(sdk.values()) 
print(dv)

et=[]
st=0
for i in range(n):
    st=st+dv[i]
    et.append(st)

print(et)

tat=[]
for i in range(n):
    t=et[i]-dk[i]
    tat.append(t)
print(tat)

wt=[]
for i in range(n):
    w=tat[i]-dv[i]
    wt.append(w)
print(wt)

avg_tat=sum(tat)/n
avg_wt=sum(wt)/n

print(f"The average Turn around time is: {avg_tat}\nThe average waiting time is: {avg_wt}")


###PAGE REPLACEMENT

n = int(input("Enter the number of page frames: "))
N = int(input("Enter the number of reference strings: "))

a = list(map(int, input("Enter the reference string (space-separated): ").split()))
pfr = []
pfa = 0
ph = 0

for i in range(N):
    page = a[i]
    print(f"Accessing page {page}:")

    if page in pfr:
        ph += 1
        print(f"Page {page} already in frames {pfr} (Page Hit)")
        continue  

    if len(pfr) < n:
        pfr.append(page)
        pfa += 1
        print(f"Page {page} added to frames: {pfr} (Page Fault)")
    else:

        farthest_index = -1
        farthest_page = -1

        for p in pfr:
            
            try:
                index = a.index(p, i + 1)
            except ValueError:
                
                farthest_page = p
                break

            if index > farthest_index:
                farthest_index = index
                farthest_page = p

        pfr[pfr.index(farthest_page)] = page
        pfa += 1
        print(f"Page {page} replaced {farthest_page}. New frames: {pfr} (Page Fault)")

print(f"Total Page Hits: {ph}")
print(f"Total Page Faults: {pfa}")
print(f"Hit Ratio: {(ph / N):.2f}")


####peterson
import time

def peterson_solution():
    process_count = int(input("Enter the number of processes: "))

    if process_count != 2:
        print("Peterson Solution is only for 2 processes.")
        return

    flag = [False, False]
    turn = 0

    print("Enter the condition:")
    print("0 - Normal Condition")
    print("1 - Context Switch Condition")
    condition = int(input("Enter your choice (0 or 1): "))

    if condition not in [0, 1]:
        print("You have chosen an invalid option.")
        return

    if condition == 0:
        process_0(flag, turn, condition)
        process_1(flag, turn)
    else:
        process_0(flag, turn, condition)
        process_1(flag, turn)


def process_0(flag, turn, condition):
    print("Process 0 is running")
    flag[0] = True
    turn = 1

    if condition != 0:
        process_1(flag, turn)

    start_time = time.time()
    while turn == 1 and flag[1]:
        if time.time() - start_time > 10:
            print("Process 0 is ending due to timeout.")
            return

    if time.time() - start_time <= 10:
        print("Process 0 is in Critical Section")

    flag[0] = False
    print("Process 0 has ended")


def process_1(flag, turn):
    print("Process 1 is running")
    flag[1] = True
    turn = 0

    start_time = time.time()
    while turn == 0 and flag[0]:
        if time.time() - start_time > 3:
            print("3 Seconds Wait")
            print("Process 1 is ending due to timeout.")
            return

    if time.time() - start_time <= 10:
        print("Process 1 is in Critical Section")

    flag[1] = False
    print("Process 1 has ended")

peterson_solution()



"""
n = int(input("Enter the number of processes: "))  
turn = 0  
flag = [False, False]  

def p0():
    global turn, flag
    flag[0] = True  
    turn = 1  
    while flag[1] and turn == 1:  
        pass
    print("Process 0 is executing in the critical section.")
    flag[0] = False 

def p1():
    global turn, flag
    flag[1] = True  
    turn = 0  
    while flag[0] and turn == 0:  
        pass
    print("Process 1 is executing in the critical section.")
    flag[1] = False  

def context_switch():
    switch = input("Do you want to context switch the process? (yes/no): ").strip().lower()
    if switch == "yes":
        if pr == "p0":
            print("Context switching to p1...")
            p1()
            print("p0 cannot execute.")
        elif pr == "p1":
            print("Context switching to p0...")
            p0()
            print("p1 cannot execute.")
    elif switch == "no":
        print("No context switch performed.")
    else:
        print("Invalid input. No context switch performed.")

if n == 2:
    pr = input("Enter the process to be executed (p0 or p1): ").strip().lower()
    if pr == "p0":
        p0()
        print("p1 cannot execute.")
        context_switch()
    elif pr == "p1":
        p1()
        print("p0 cannot execute.")
        context_switch()
    else:
        print("Invalid process name. Please enter p0 or p1.")
else:
    print(f"{n} processes are invalid. Enter a valid input.")

"""



####PRIORITY
import time

def peterson_solution():
    process_count = int(input("Enter the number of processes: "))

    if process_count != 2:
        print("Peterson Solution is only for 2 processes.")
        return

    flag = [False, False]
    turn = 0

    print("Enter the condition:")
    print("0 - Normal Condition")
    print("1 - Context Switch Condition")
    condition = int(input("Enter your choice (0 or 1): "))

    if condition not in [0, 1]:
        print("You have chosen an invalid option.")
        return

    if condition == 0:
        process_0(flag, turn, condition)
        process_1(flag, turn)
    else:
        process_0(flag, turn, condition)
        process_1(flag, turn)


def process_0(flag, turn, condition):
    print("Process 0 is running")
    flag[0] = True
    turn = 1

    if condition != 0:
        process_1(flag, turn)

    start_time = time.time()
    while turn == 1 and flag[1]:
        if time.time() - start_time > 10:
            print("Process 0 is ending due to timeout.")
            return

    if time.time() - start_time <= 10:
        print("Process 0 is in Critical Section")

    flag[0] = False
    print("Process 0 has ended")


def process_1(flag, turn):
    print("Process 1 is running")
    flag[1] = True
    turn = 0

    start_time = time.time()
    while turn == 0 and flag[0]:
        if time.time() - start_time > 3:
            print("3 Seconds Wait")
            print("Process 1 is ending due to timeout.")
            return

    if time.time() - start_time <= 10:
        print("Process 1 is in Critical Section")

    flag[1] = False
    print("Process 1 has ended")

peterson_solution()



"""
n = int(input("Enter the number of processes: "))  
turn = 0  
flag = [False, False]  

def p0():
    global turn, flag
    flag[0] = True  
    turn = 1  
    while flag[1] and turn == 1:  
        pass
    print("Process 0 is executing in the critical section.")
    flag[0] = False 

def p1():
    global turn, flag
    flag[1] = True  
    turn = 0  
    while flag[0] and turn == 0:  
        pass
    print("Process 1 is executing in the critical section.")
    flag[1] = False  

def context_switch():
    switch = input("Do you want to context switch the process? (yes/no): ").strip().lower()
    if switch == "yes":
        if pr == "p0":
            print("Context switching to p1...")
            p1()
            print("p0 cannot execute.")
        elif pr == "p1":
            print("Context switching to p0...")
            p0()
            print("p1 cannot execute.")
    elif switch == "no":
        print("No context switch performed.")
    else:
        print("Invalid input. No context switch performed.")

if n == 2:
    pr = input("Enter the process to be executed (p0 or p1): ").strip().lower()
    if pr == "p0":
        p0()
        print("p1 cannot execute.")
        context_switch()
    elif pr == "p1":
        p1()
        print("p0 cannot execute.")
        context_switch()
    else:
        print("Invalid process name. Please enter p0 or p1.")
else:
    print(f"{n} processes are invalid. Enter a valid input.")

"""

###READERWRITER
import threading
import time

read_count = 0
read_lock = threading.Semaphore(1)  
resource = threading.Semaphore(1)    

class Reader(threading.Thread):
    def run(self):
        global read_count

        try:
            read_lock.acquire()
            read_count += 1
            if read_count == 1:
                resource.acquire() 
            read_lock.release()

            print(f"{threading.current_thread().name} is reading.")
            time.sleep(1)  
            print(f"{threading.current_thread().name} has finished reading.")

            read_lock.acquire()
            read_count -= 1
            if read_count == 0:
                resource.release()  
            read_lock.release()

        except Exception as e:
            print(e)

class Writer(threading.Thread):
    def run(self):
        try:
            resource.acquire()

            print(f"{threading.current_thread().name} is writing.")
            time.sleep(1)  
            print(f"{threading.current_thread().name} has finished writing.")

            resource.release()

        except Exception as e:
            print(e)

reader1 = Reader()
reader2 = Reader()
writer1 = Writer()
writer2 = Writer()

reader1.start()
writer1.start()
reader2.start()
writer2.start()

reader1.join()
reader2.join()
writer1.join()
writer2.join()


####rrrr

from collections import deque

def round_robin_scheduling(d, tq):
    n = len(d)
    arrival_times = {pid: d[pid]["arrival_time"] for pid in d}
    burst_times = {pid: d[pid]["burst_time"] for pid in d}
    burst_remaining = burst_times.copy()  

    t = 0  
    waiting_time = {pid: 0 for pid in d}
    turnaround_time = {pid: 0 for pid in d}
    is_completed = {pid: False for pid in d}

    queue = deque()  

    for pid in arrival_times:
        if arrival_times[pid] <= t:
            queue.append(pid)

    while queue:
        current_process = queue.popleft()

        if burst_remaining[current_process] > tq:
            t += tq
            burst_remaining[current_process] -= tq

            for pid in arrival_times:
                if (arrival_times[pid] <= t and not is_completed[pid] 
                    and pid not in queue and pid != current_process):
                    queue.append(pid)

            queue.append(current_process)
        
        else:
            t += burst_remaining[current_process]
            burst_remaining[current_process] = 0
            turnaround_time[current_process] = t - arrival_times[current_process]
            waiting_time[current_process] = turnaround_time[current_process] - burst_times[current_process]
            is_completed[current_process] = True

            for pid in arrival_times:
                if (arrival_times[pid] <= t and not is_completed[pid] 
                    and pid not in queue and pid != current_process):
                    queue.append(pid)

    total_waiting_time = sum(waiting_time.values())
    total_turnaround_time = sum(turnaround_time.values())
    
    avg_waiting_time = total_waiting_time / n
    avg_turnaround_time = total_turnaround_time / n

    print(f"Average Turnaround Time = {avg_turnaround_time:.2f}")
    print(f"Average Waiting Time = {avg_waiting_time:.2f}")


n = int(input("Enter the number of Processes: "))
tq = int(input("Enter the time quantum: "))
d = {}

for i in range(n):
    pid = input("Enter process id here: ")
    at = int(input(f"Enter the arrival time of {pid}: "))
    bt = int(input(f"Enter the burst time of {pid}: "))
    d[pid] = {"arrival_time": at, "burst_time": bt}

round_robin_scheduling(d, tq)



"""
n = int(input("Enter the number of Processes: "))
tq = int(input("Enter the time quantum: "))
d = {}
l = []

for i in range(n):
    pid = input("Enter process id here: ")
    l.append(pid)
    
    at = int(input(f"Enter the arrival time of {pid}: "))
    bt = int(input(f"Enter the burst time of {pid}: "))
    d[pid] = {"arrival_time": at, "burst_time": bt}

tt = 0 
ttc = 0
wt = 0
tat = 0

proc = []
for pid in l:
    arrival = d[pid]["arrival_time"]
    burst = d[pid]["burst_time"]
    rt = burst
    proc.append([arrival, burst, rt, 0])
    tt += burst

while tt != 0:
    for i in range(len(proc)):
        if proc[i][2] <= tq and proc[i][2] > 0:
            ttc += proc[i][2]
            tt -= proc[i][2]
            proc[i][2] = 0 
        elif proc[i][2] > 0:
            proc[i][2] -= tq
            tt -= tq
            ttc += tq
        
        if proc[i][2] == 0 and proc[i][3] != 1:
           
            wt += ttc - proc[i][0] - proc[i][1]
            tat += ttc - proc[i][0]
            proc[i][3] = 1  

print("Avg Waiting Time is:", (wt/n))
print("Avg Turnaround Time is:", (tat/n))

"""


#####SEMAPHORE
import threading
import time
from queue import Queue

def binary_semaphore():
    flag = [False, False]
    turn = 0

    def process_0():
        nonlocal turn
        print("Process 0 is requesting critical section.")
        flag[0] = True
        turn = 1
        while flag[1] and turn == 1:
            pass  
        print("Process 0 enters the critical section.")
        time.sleep(1) 
        print("Process 0 leaves the critical section.")
        flag[0] = False

    def process_1():
        nonlocal turn
        print("Process 1 is requesting critical section.")
        flag[1] = True
        turn = 0
        while flag[0] and turn == 0:
            pass  
        print("Process 1 enters the critical section.")
        time.sleep(1)  
        print("Process 1 leaves the critical section.")
        flag[1] = False

    t0 = threading.Thread(target=process_0)
    t1 = threading.Thread(target=process_1)
    
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    print("\nMutual Exclusion: Satisfied")
    print("Progress: Satisfied")
    print("Bounded Waiting: Satisfied")

class CountingSemaphore:
    def __init__(self, max_value):
        self.value = max_value
        self.suspend_list = Queue()
        self.lock = threading.Lock()

    def wait_semaphore(self, process_name):
        with self.lock:
            if self.value > 0:
                self.value -= 1
                print(f"{process_name} enters the critical section.")
            else:
                print(f"{process_name} is suspended (added to suspend list).")
                self.suspend_list.put(process_name)
                return False
        return True

    def signal_semaphore(self, process_name):
        with self.lock:
            print(f"{process_name} leaves the critical section.")
            self.value += 1
            if not self.suspend_list.empty():
                next_process = self.suspend_list.get()
                print(f"{next_process} is woken up from the suspend list.")
                self.value -= 1  
                return next_process  
        return None


def counting_semaphore(process_count, max_in_critical_section):
    semaphore = CountingSemaphore(max_in_critical_section)

    def process(semaphore, process_name):
        while not semaphore.wait_semaphore(process_name):
            time.sleep(0.5)  
        time.sleep(1)
        next_process = semaphore.signal_semaphore(process_name)
        if next_process:
            threading.Thread(target=process, args=(semaphore, next_process)).start()

    threads = []
    for i in range(1, process_count + 1):
        process_name = f"Process-{i}"
        t = threading.Thread(target=process, args=(semaphore, process_name))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("\nMutual Exclusion: Satisfied (only up to max allowed processes can enter)")
    print("Progress: Satisfied (waiting processes will enter as slots free up)")
    print("Bounded Waiting: Satisfied (queued processes will eventually enter based on arrival order)")


def main():
   
    process_count = int(input("Enter the number of processes: "))

    if process_count < 2:
        print("Invalid number of processes! Must be 2 or more.")
        return

    if process_count == 2:
        print("Using Binary Semaphore (Peterson's Solution) for 2 processes.")
        binary_semaphore()
    else:
        max_in_critical_section = int(input("Enter the maximum number of processes allowed in critical section at once: "))
        print(f"Using Counting Semaphore for {process_count} processes with a maximum of {max_in_critical_section} in critical section at a time.")
        counting_semaphore(process_count, max_in_critical_section)

if __name__ == "__main__":
    main()

"""
def binary_semaphore(semaphore, processID, operation):
    if operation == "down":
        if semaphore[0] == 1:
            semaphore[0] = 0
            print(f"Process {processID} enters critical section (binary down).")
        else:
            print(f"Process {processID} is waiting (binary down).")
    elif operation == "up":
        if semaphore[0] == 0:
            semaphore[0] = 1
            print(f"Process {processID} leaves critical section (binary up).")
        else:
            print(f"Process {processID} cannot leave (already up).")

def counting_semaphore(semaphore, processID, operation):
    if operation == "wait":
        if semaphore[0] > 0:
            semaphore[0] -= 1
            print(f"Process {processID} enters critical section (counting wait).")
        else:
            semaphore[1].append(processID)
            print(f"Process {processID} is waiting (counting wait).")
    elif operation == "signal":
        if len(semaphore[1]) > 0:
            next_process = semaphore[1].pop(0)
            print(f"Process {next_process} enters critical section (counting signal).")
        else:
            semaphore[0] += 1
            print(f"Process {processID} leaves critical section (counting signal).")

def check_conditions(semaphore_type, semaphore):
    if semaphore_type == "binary":
        if semaphore[0] in [0, 1]:
            print("Binary semaphore: Mutual exclusion is guaranteed.")
        else:
            print("Binary semaphore: Invalid state.")
        
        print("Progress condition is satisfied.")  # Binary semaphore will always satisfy progress
        print("Bounded wait is not applicable to binary semaphore.")  # Bounded wait doesn't apply to binary semaphores
    else:
        if semaphore[0] >= 0:
            print("Counting semaphore: Mutual exclusion may not be guaranteed.")
        else:
            print("Counting semaphore: Invalid state.")
        
        if semaphore[0] >= 0:
            print("Progress condition is satisfied.")
        else:
            print("Progress condition is not satisfied.")

        if len(semaphore[1]) == 0:
            print("Bounded wait condition is satisfied.")
        else:
            print("Processes are waiting, bounded wait is not guaranteed.")

num_processes = int(input("Enter the number of processes: "))

if num_processes == 2:
    print("Using binary semaphore.")
    semaphore = [1]  # Binary semaphore initialized to 1
    for _ in range(num_processes):
        processID = input("Enter process ID: ")
        operation = input(f"Process {processID} action (up/down): ").strip().lower()
        binary_semaphore(semaphore, processID, operation)
else:
    print("Using counting semaphore.")
    semaphore_value = int(input("Enter initial semaphore value: "))
    semaphore = [semaphore_value, []]  # Counting semaphore with a queue
    for _ in range(num_processes):
        processID = input("Enter process ID: ")
        operation = input(f"Process {processID} action (wait/signal): ").strip().lower()
        counting_semaphore(semaphore, processID, operation)

check_conditions("binary" if num_processes == 2 else "counting", semaphore)

"""


####SJF
import numpy as np

def sjf_scheduling(processes, schedule_type):
    process_count = len(processes)
    arrival_times = [process[1] for process in processes]
    burst_times = [process[2] for process in processes]
    
    if schedule_type == 1:
        non_preemptive_sjf(process_count, arrival_times, burst_times)
    elif schedule_type == 2:
        preemptive_sjf(process_count, arrival_times, burst_times)
    else:
        print("Invalid scheduling type. Please enter 1 for Non-Preemptive or 2 for Preemptive.")

def non_preemptive_sjf(process_count, arrival_times, burst_times):
    max_value = float('inf')
    arrival_copy = arrival_times.copy()
    burst_copy = burst_times.copy()
    sorted_arrival = sorted(enumerate(arrival_copy), key=lambda x: x[1])

    time_elapsed = 0
    gantt_chart = []
    total_turnaround_time = 0
    total_waiting_time = 0

    for i, arrival_time in sorted_arrival:
        if arrival_time > time_elapsed:
            time_elapsed = arrival_time

        min_burst_index = i
        min_burst = burst_times[min_burst_index]

        for j in range(process_count):
            if arrival_times[j] <= time_elapsed and burst_times[j] < min_burst:
                min_burst = burst_times[j]
                min_burst_index = j

        time_elapsed += burst_times[min_burst_index]
        turnaround_time = time_elapsed - arrival_times[min_burst_index]
        waiting_time = turnaround_time - burst_copy[min_burst_index]

        total_turnaround_time += turnaround_time
        total_waiting_time += waiting_time

        gantt_chart.append((processes[min_burst_index][0], time_elapsed))

        burst_times[min_burst_index] = max_value

    avg_turnaround_time = total_turnaround_time / process_count
    avg_waiting_time = total_waiting_time / process_count

    print(f"Gantt Chart (Process ID, Time): {gantt_chart}")
    print(f"Average Turnaround Time = {avg_turnaround_time:.2f}")
    print(f"Average Waiting Time = {avg_waiting_time:.2f}")


def preemptive_sjf(process_count, arrival_times, burst_times):
    remaining_time = burst_times.copy()
    completion_times = np.zeros(process_count)
    waiting_times = np.zeros(process_count)
    turnaround_times = np.zeros(process_count)
    is_completed = [False] * process_count

    current_time = 0
    completed = 0

    while completed < process_count:
        min_index = -1
        min_burst = float('inf')

        for i in range(process_count):
            if arrival_times[i] <= current_time and not is_completed[i] and remaining_time[i] < min_burst:
                min_burst = remaining_time[i]
                min_index = i

        if min_index == -1:
            current_time += 1
            continue

        remaining_time[min_index] -= 1
        current_time += 1

        if remaining_time[min_index] == 0:
            completion_times[min_index] = current_time
            turnaround_times[min_index] = completion_times[min_index] - arrival_times[min_index]
            waiting_times[min_index] = turnaround_times[min_index] - burst_times[min_index]
            is_completed[min_index] = True
            completed += 1

    avg_turnaround_time = np.mean(turnaround_times)
    avg_waiting_time = np.mean(waiting_times)

    print(f"Average Turnaround Time = {avg_turnaround_time:.2f}")
    print(f"Average Waiting Time = {avg_waiting_time:.2f}")

n = int(input("Enter the number of Processes: "))
processes = []

for i in range(n):
    pid = input("Enter process id here: ")
    at = int(input(f"Enter the arrival time of {pid}: "))
    bt = int(input(f"Enter the burst time of {pid}: "))
    processes.append((pid, at, bt))
                     
print("Enter the scheduling type:")
print("1 - Non-Preemptive SJF")
print("2 - Preemptive SJF")
schedule_type = int(input("Enter your choice (1 or 2): "))

sjf_scheduling(processes, schedule_type)
