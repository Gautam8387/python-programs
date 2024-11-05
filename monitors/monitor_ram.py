import threading
import time

import psutil


# Function to track RAM usage
def ram_usage(ram_used, stop_flag, interval=1):
    process = psutil.Process()
    while not stop_flag.is_set():
        ram_used.append(process.memory_info().rss / (1024**2))  # Convert to MB
        time.sleep(interval)


# Function to run scarf.CrDirReader
def run_function(fun_check, stop_flag, *args, **kwargs):
    _ = fun_check(*args, **kwargs)
    stop_flag.set()  # Signal to stop monitoring RAM usage when the function finishes


def monitor_ram(fun_check, *args, **kwargs):
    # List to store RAM usage data
    ram_used = []
    stop_flag = threading.Event()

    # Create threads for both RAM monitoring and running the function
    monitor_thread = threading.Thread(target=ram_usage, args=(ram_used, stop_flag))
    reader_thread = threading.Thread(
        target=run_function, args=(fun_check, stop_flag, *args), kwargs=kwargs
    )

    # Start both threads
    monitor_thread.start()
    reader_thread.start()

    # Wait for the function to complete
    reader_thread.join()

    # Stop the RAM monitoring
    stop_flag.set()
    monitor_thread.join()

    return ram_used
