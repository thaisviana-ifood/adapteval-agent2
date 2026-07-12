"""Queue management for asynchronous processing"""

from typing import Any, Callable, Optional, List
from abc import ABC, abstractmethod
import asyncio

from src.shared.logger import get_logger

logger = get_logger(__name__)


class Queue(ABC):
    """Abstract base class for queue implementations"""

    @abstractmethod
    async def enqueue(self, task: Any) -> bool:
        """Add task to queue"""
        pass

    @abstractmethod
    async def dequeue(self) -> Optional[Any]:
        """Remove and return task from queue"""
        pass

    @abstractmethod
    async def size(self) -> int:
        """Get queue size"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all tasks"""
        pass


class InMemoryQueue(Queue):
    """Simple in-memory queue for development"""

    def __init__(self):
        self.tasks: List[Any] = []

    async def enqueue(self, task: Any) -> bool:
        """Add task to queue"""
        self.tasks.append(task)
        logger.debug(f"Enqueued task. Queue size: {len(self.tasks)}")
        return True

    async def dequeue(self) -> Optional[Any]:
        """Remove and return task from queue"""
        if not self.tasks:
            return None
        task = self.tasks.pop(0)
        logger.debug(f"Dequeued task. Queue size: {len(self.tasks)}")
        return task

    async def size(self) -> int:
        """Get queue size"""
        return len(self.tasks)

    async def clear(self) -> bool:
        """Clear all tasks"""
        self.tasks.clear()
        logger.info("Queue cleared")
        return True


class TaskProcessor:
    """Process tasks from queue"""

    def __init__(
        self,
        queue: Queue,
        worker_count: int = 4,
    ):
        self.queue = queue
        self.worker_count = worker_count
        self.is_running = False

    async def process_task(
        self, task: Any, handler: Callable
    ) -> Any:
        """Process a single task"""
        try:
            result = await handler(task)
            logger.debug(f"Task processed successfully")
            return result
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            return None

    async def run(self, handler: Callable) -> None:
        """Run worker loop"""
        self.is_running = True
        workers = [
            asyncio.create_task(self._worker(handler))
            for _ in range(self.worker_count)
        ]
        await asyncio.gather(*workers)

    async def _worker(self, handler: Callable) -> None:
        """Individual worker coroutine"""
        while self.is_running:
            task = await self.queue.dequeue()
            if task:
                await self.process_task(task, handler)
            else:
                await asyncio.sleep(0.1)

    def stop(self) -> None:
        """Stop processing"""
        self.is_running = False
        logger.info("Task processor stopped")
