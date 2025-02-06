import multiprocessing
from multiprocessing import Process, Queue
import queue
import time

class Parallel_Task_Process:
    def __init__(self, target, args=(), kwargs=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs.copy() if kwargs else {}
        self._result_queue = Queue()  # For passing results/exceptions
        self._process = Process(
            target=Parallel_Task_Process._run,
            args=(self.target, self.args, self.kwargs, self._result_queue)
        )

    def add_kwargs(self, **kwargs):
        """Add or update keyword arguments for the task."""
        self.kwargs.update(kwargs)
        pass

    def launch(self):
        """Start the task in a separate process."""
        self._process.start()

    @staticmethod
    def _run(target, args, kwargs, result_queue):
        """Internal method executed in the process."""
        try:
            result = target(*args, **kwargs)
            result_queue.put(('result', result))
        except Exception as e:
            result_queue.put(('error', e))

    def is_completed(self):
        """Check if the task process has completed."""
        return not self._process.is_alive()

    def get_result(self):
        """
        Retrieve the result or exception from the task.
        Raises `RuntimeError` if the task was terminated before completion.
        """
        if not self.is_completed():
            raise ValueError("Task has not completed yet.")
        
        try:
            type_, value = self._result_queue.get_nowait()
            if type_ == 'error':
                raise value  # Re-raise the exception from the task
            return value
        except queue.Empty:
            raise RuntimeError("Task was terminated before completion.")

    def stop(self):
        """Forcefully terminate the task process."""
        if self._process.is_alive():
            self._process.terminate()  # Force-stop the process
            self._process.join()  # Cleanup process resources

    def wait_for_completion(self, timeout=None):
        """Wait for the task to finish, with optional timeout."""
        self._process.join(timeout)
        return self.is_completed()

import threading
import inspect
import time

class Parallel_Task_Thread:
    def __init__(self, target, args=(), kwargs=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs.copy() if kwargs else {}
        self._stop_event = threading.Event()
        self._thread = None
        self._result = None
        self._exception = None
        self._completed = False

    def add_kwargs(self, **kwargs):
        """Add or update keyword arguments for the task."""
        self.kwargs.update(kwargs)
        pass

    def launch(self):
        """Start the task in a separate thread."""
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _run(self):
        """Internal method executed in the thread, runs the target function."""
        try:
            kwargs = self.kwargs.copy()
            # Check if the target function accepts '_stop_event' as a parameter
            sig = inspect.signature(self.target)
            if '_stop_event' in sig.parameters:
                kwargs['_stop_event'] = self._stop_event
            self._result = self.target(*self.args, **kwargs)
        except Exception as e:
            self._exception = e
        finally:
            self._completed = True

    def is_completed(self):
        """Check if the task has completed."""
        return self._completed

    def get_result(self):
        """
        Retrieve the result of the task. If the task raised an exception,
        it will be re-raised here.
        """
        if not self._completed:
            raise ValueError("Task has not completed yet.")
        if self._exception:
            raise self._exception
        return self._result

    def stop(self):
        """Request the task to stop by setting the stop event."""
        self._stop_event.set()

    def wait_for_completion(self, timeout=None):
        """
        Wait for the task to complete, optionally with a timeout.
        Returns True if the task completed, False otherwise.
        """
        self._thread.join(timeout)
        return self._completed    
