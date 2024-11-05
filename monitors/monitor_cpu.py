import threading
import time

import psutil


# Function to track CPU usage
def cpu_usage(cpu_used, stop_flag, interval=1):
    process = psutil.Process()
    while not stop_flag.is_set():
        cpu_used.append(process.cpu_percent(interval=interval))  # CPU usage percentage
        time.sleep(interval)


# Function to run the target function
def run_function(fun_check, stop_flag, *args, **kwargs):
    _ = fun_check(*args, **kwargs)
    stop_flag.set()  # Signal to stop monitoring CPU usage when the function finishes


def monitor_cpu(fun_check, *args, **kwargs):
    # List to store CPU usage data
    cpu_used = []
    stop_flag = threading.Event()

    # Create threads for both CPU monitoring and running the function
    monitor_thread = threading.Thread(target=cpu_usage, args=(cpu_used, stop_flag))
    reader_thread = threading.Thread(
        target=run_function, args=(fun_check, stop_flag, *args), kwargs=kwargs
    )

    # Start both threads
    monitor_thread.start()
    reader_thread.start()

    # Wait for the function to complete
    reader_thread.join()

    # Stop the CPU monitoring
    stop_flag.set()
    monitor_thread.join()

    return cpu_used
