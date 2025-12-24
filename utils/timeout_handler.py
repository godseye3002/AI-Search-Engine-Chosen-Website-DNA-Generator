"""
Timeout Handler Utility for GodsEye Pipeline

Provides timeout protection for long-running functions with proper
exception handling and logging.
"""

import signal
import threading
import time
import logging
from typing import Any, Callable, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TimeoutResult(Enum):
    """Result of timeout execution"""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Result of function execution with timeout"""
    result: Any
    status: TimeoutResult
    execution_time: float
    error_message: Optional[str] = None
    timeout_duration: Optional[float] = None


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


class TimeoutHandler:
    """
    Handles function execution with timeout protection.
    
    Supports both threading-based and signal-based timeout mechanisms
    depending on the platform and use case.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def execute_with_timeout(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        timeout: float = 30.0,
        thread_based: bool = True
    ) -> ExecutionResult:
        """
        Execute a function with timeout protection.
        
        Args:
            func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            timeout: Timeout in seconds
            thread_based: Use threading-based timeout (recommended)
            
        Returns:
            ExecutionResult with result, status, and timing info
            
        Example:
            >>> handler = TimeoutHandler()
            >>> result = handler.execute_with_timeout(
            ...     long_running_function, 
            ...     args=(arg1, arg2),
            ...     timeout=10.0
            ... )
            >>> if result.status == TimeoutResult.SUCCESS:
            ...     print(f"Result: {result.result}")
            >>> elif result.status == TimeoutResult.TIMEOUT:
            ...     print("Function timed out")
        """
        if kwargs is None:
            kwargs = {}
        
        start_time = time.time()
        
        if thread_based:
            return self._execute_with_thread_timeout(func, args, kwargs, timeout, start_time)
        else:
            return self._execute_with_signal_timeout(func, args, kwargs, timeout, start_time)
    
    def _execute_with_thread_timeout(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        timeout: float,
        start_time: float
    ) -> ExecutionResult:
        """Execute function using threading-based timeout"""
        
        result_container = {}
        exception_container = {}
        
        def target():
            try:
                result_container['result'] = func(*args, **kwargs)
            except Exception as e:
                exception_container['exception'] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        execution_time = time.time() - start_time
        
        if thread.is_alive():
            # Thread is still running -> timeout
            self.logger.warning(f"Function timed out after {timeout:.2f} seconds")
            return ExecutionResult(
                result=None,
                status=TimeoutResult.TIMEOUT,
                execution_time=execution_time,
                error_message=f"Function timed out after {timeout:.2f} seconds",
                timeout_duration=timeout
            )
        
        # Thread completed
        if exception_container:
            error = exception_container['exception']
            self.logger.error(f"Function raised exception: {str(error)}")
            return ExecutionResult(
                result=None,
                status=TimeoutResult.ERROR,
                execution_time=execution_time,
                error_message=str(error)
            )
        
        # Success
        result = result_container.get('result')
        self.logger.debug(f"Function completed successfully in {execution_time:.2f} seconds")
        return ExecutionResult(
            result=result,
            status=TimeoutResult.SUCCESS,
            execution_time=execution_time
        )
    
    def _execute_with_signal_timeout(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        timeout: float,
        start_time: float
    ) -> ExecutionResult:
        """Execute function using signal-based timeout (Unix only)"""
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function timed out after {timeout} seconds")
        
        # Set up signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        
        try:
            signal.alarm(int(timeout))
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel alarm
            
            execution_time = time.time() - start_time
            self.logger.debug(f"Function completed successfully in {execution_time:.2f} seconds")
            
            return ExecutionResult(
                result=result,
                status=TimeoutResult.SUCCESS,
                execution_time=execution_time
            )
            
        except TimeoutError as e:
            execution_time = time.time() - start_time
            self.logger.warning(f"Function timed out: {str(e)}")
            
            return ExecutionResult(
                result=None,
                status=TimeoutResult.TIMEOUT,
                execution_time=execution_time,
                error_message=str(e),
                timeout_duration=timeout
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Function raised exception: {str(e)}")
            
            return ExecutionResult(
                result=None,
                status=TimeoutResult.ERROR,
                execution_time=execution_time,
                error_message=str(e)
            )
            
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGALRM, old_handler)
            signal.alarm(0)  # Ensure alarm is cancelled
    
    def execute_batch_with_timeout(
        self,
        func_list: list,
        timeout_per_item: float,
        max_parallel: int = None
    ) -> list:
        """
        Execute multiple functions in parallel with individual timeouts.
        
        Args:
            func_list: List of (func, args, kwargs) tuples
            timeout_per_item: Timeout for each function
            max_parallel: Maximum parallel executions
            
        Returns:
            List of ExecutionResult objects
        """
        if max_parallel is None:
            max_parallel = len(func_list)
        
        results = []
        threads = []
        result_containers = [{} for _ in func_list]
        
        def execute_item(index, func_args_kwargs):
            func, args, kwargs = func_args_kwargs
            result = self.execute_with_timeout(
                func, args, kwargs, timeout_per_item
            )
            result_containers[index]['result'] = result
        
        # Process in batches to control parallelism
        for i in range(0, len(func_list), max_parallel):
            batch = func_list[i:i + max_parallel]
            batch_threads = []
            
            for j, (func, args, kwargs) in enumerate(batch):
                thread = threading.Thread(
                    target=execute_item,
                    args=(i + j, (func, args, kwargs))
                )
                thread.daemon = True
                thread.start()
                batch_threads.append(thread)
            
            # Wait for batch to complete
            for thread in batch_threads:
                thread.join()
            
            # Collect results
            for container in result_containers[i:i + max_parallel]:
                if 'result' in container:
                    results.append(container['result'])
                else:
                    # Shouldn't happen, but just in case
                    results.append(ExecutionResult(
                        result=None,
                        status=TimeoutResult.ERROR,
                        execution_time=0,
                        error_message="Unknown error during batch execution"
                    ))
        
        return results


# Global instance for convenience
default_timeout_handler = TimeoutHandler()

def execute_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    timeout: float = 30.0,
    thread_based: bool = True
) -> ExecutionResult:
    """
    Convenience function for timeout execution.
    
    This is the main function that should be used throughout the pipeline.
    
    Args:
        func: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments  
        timeout: Timeout in seconds
        thread_based: Use threading-based timeout
        
    Returns:
        ExecutionResult object
    """
    return default_timeout_handler.execute_with_timeout(
        func, args, kwargs, timeout, thread_based
    )


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    def quick_function(x, y):
        """Quick function for testing"""
        time.sleep(1)
        return x + y
    
    def slow_function():
        """Slow function that will timeout"""
        time.sleep(5)
        return "This should timeout"
    
    def error_function():
        """Function that raises an error"""
        raise ValueError("Test error")
    
    handler = TimeoutHandler()
    
    print("=== Timeout Handler Test ===")
    
    # Test 1: Successful execution
    print("\nTest 1: Quick function (should succeed)")
    result1 = handler.execute_with_timeout(quick_function, args=(2, 3), timeout=3)
    print(f"Status: {result1.status}, Result: {result1.result}, Time: {result1.execution_time:.2f}s")
    
    # Test 2: Timeout
    print("\nTest 2: Slow function (should timeout)")
    result2 = handler.execute_with_timeout(slow_function, timeout=2)
    print(f"Status: {result2.status}, Error: {result2.error_message}, Time: {result2.execution_time:.2f}s")
    
    # Test 3: Error handling
    print("\nTest 3: Error function (should handle error)")
    result3 = handler.execute_with_timeout(error_function, timeout=3)
    print(f"Status: {result3.status}, Error: {result3.error_message}, Time: {result3.execution_time:.2f}s")
    
    # Test 4: Batch execution
    print("\nTest 4: Batch execution")
    func_list = [
        (quick_function, (1, 2), {}),
        (quick_function, (3, 4), {}),
        (slow_function, (), {}),
        (error_function, (), {})
    ]
    results4 = handler.execute_batch_with_timeout(func_list, timeout_per_item=2, max_parallel=2)
    for i, res in enumerate(results4):
        print(f"Batch item {i}: Status={res.status}, Time={res.execution_time:.2f}s")
