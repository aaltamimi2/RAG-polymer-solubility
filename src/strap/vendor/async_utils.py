"""
Async Utilities for Polymer Solubility Agent

Provides thread pool execution for blocking operations and async conversion utilities.
Used to make synchronous operations non-blocking in async context.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

# Thread pool for blocking operations
# 4 workers is suitable for I/O-bound tasks (DB queries, file I/O)
_thread_pool = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="agent_worker"
)


async def run_in_thread(func: Callable, *args, **kwargs) -> Any:
    """
    Run a synchronous function in a thread pool to avoid blocking the event loop.

    This is essential for:
    - Database queries (DuckDB operations)
    - File I/O (matplotlib savefig)
    - CPU-intensive operations

    Usage:
        result = await run_in_thread(blocking_function, arg1, arg2, kwarg=value)

    Args:
        func: Synchronous function to execute
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from func execution
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_thread_pool, lambda: func(*args, **kwargs))


def async_wrapper(sync_func: Callable) -> Callable:
    """
    Decorator to convert a synchronous function to async.

    The wrapped function will execute in a thread pool when awaited,
    preventing it from blocking the event loop.

    Usage:
        @async_wrapper
        def blocking_function(arg1, arg2):
            # ... blocking code ...
            return result

        # Can now be called with await
        result = await blocking_function(val1, val2)

    Args:
        sync_func: Synchronous function to wrap

    Returns:
        Async function that executes sync_func in thread pool
    """
    @wraps(sync_func)
    async def wrapper(*args, **kwargs):
        return await run_in_thread(sync_func, *args, **kwargs)
    return wrapper


def shutdown_thread_pool():
    """
    Shutdown the thread pool gracefully.

    Call this during application shutdown to ensure all threads complete.
    """
    global _thread_pool
    if _thread_pool:
        logger.info("Shutting down thread pool...")
        _thread_pool.shutdown(wait=True)
        logger.info("Thread pool shutdown complete")
