"""
Parallel agent coordination for the RFQ system.

This module implements parallel execution patterns following Anthropic's
multi-agent research, enabling simultaneous agent execution for improved
performance and throughput.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, TypeVar
from dataclasses import dataclass
from datetime import datetime

from ...core.interfaces.agent import BaseAgent, AgentContext, AgentStatus
from ...core.models.rfq import RFQProcessingResult


T = TypeVar('T')
R = TypeVar('R')


@dataclass
class ParallelTask:
    """Represents a task to be executed in parallel."""
    task_id: str
    agent: BaseAgent
    input_data: Any
    context: AgentContext
    priority: int = 1  # 1=highest, 5=lowest
    timeout_seconds: float = 30.0
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class TaskResult:
    """Result of a parallel task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    agent_id: str = ""
    retry_count: int = 0


class ParallelExecutionError(Exception):
    """Exception raised during parallel execution."""
    pass


class ParallelCoordinator:
    """
    Coordinates parallel execution of multiple agents.
    
    This coordinator implements Anthropic's pattern of distributing work
    across agents with separate context windows to add more capacity
    for parallel reasoning.
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        default_timeout: float = 30.0,
        enable_health_monitoring: bool = True
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        self.enable_health_monitoring = enable_health_monitoring
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._task_results: Dict[str, TaskResult] = {}
        self._execution_stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0.0,
            "peak_concurrent_tasks": 0
        }
    
    async def execute_parallel_tasks(
        self, 
        tasks: List[ParallelTask],
        wait_for_all: bool = True,
        return_on_first_success: bool = False
    ) -> Dict[str, TaskResult]:
        """
        Execute multiple tasks in parallel.
        
        Args:
            tasks: List of tasks to execute
            wait_for_all: Whether to wait for all tasks to complete
            return_on_first_success: Return as soon as one task succeeds
            
        Returns:
            Dictionary mapping task_id to TaskResult
        """
        if not tasks:
            return {}
        
        # Sort tasks by priority (1=highest priority)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority)
        
        # Create semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # Create coroutines for each task
        coroutines = [
            self._execute_single_task(task, semaphore)
            for task in sorted_tasks
        ]
        
        start_time = time.time()
        results = {}
        
        try:
            if return_on_first_success:
                # Return as soon as one task succeeds
                results = await self._execute_until_first_success(coroutines, sorted_tasks)
            elif wait_for_all:
                # Wait for all tasks to complete
                task_results = await asyncio.gather(*coroutines, return_exceptions=True)
                results = self._process_gather_results(task_results, sorted_tasks)
            else:
                # Start all tasks but don't wait for completion
                for i, coro in enumerate(coroutines):
                    task_id = sorted_tasks[i].task_id
                    self._active_tasks[task_id] = asyncio.create_task(coro)
                results = {task.task_id: TaskResult(
                    task_id=task.task_id,
                    success=False,
                    agent_id=task.agent.agent_id
                ) for task in sorted_tasks}
        
        except Exception as e:
            # Handle coordination-level errors
            for task in sorted_tasks:
                if task.task_id not in results:
                    results[task.task_id] = TaskResult(
                        task_id=task.task_id,
                        success=False,
                        error=f"Coordination error: {str(e)}",
                        agent_id=task.agent.agent_id
                    )
        
        # Update execution statistics
        execution_time = (time.time() - start_time) * 1000
        self._update_execution_stats(results, execution_time)
        
        return results
    
    async def _execute_single_task(
        self, 
        task: ParallelTask, 
        semaphore: asyncio.Semaphore
    ) -> TaskResult:
        """Execute a single task with proper error handling and metrics."""
        async with semaphore:
            start_time = time.time()
            
            # Update peak concurrent tasks
            current_active = len([t for t in self._active_tasks.values() if not t.done()])
            self._execution_stats["peak_concurrent_tasks"] = max(
                self._execution_stats["peak_concurrent_tasks"],
                current_active + 1
            )
            
            try:
                # Health check before execution if enabled
                if self.enable_health_monitoring:
                    health = await task.agent.health_check()
                    if health.status != AgentStatus.HEALTHY:
                        return TaskResult(
                            task_id=task.task_id,
                            success=False,
                            error=f"Agent unhealthy: {health.status}",
                            agent_id=task.agent.agent_id
                        )
                
                # Execute the task with timeout
                result = await asyncio.wait_for(
                    task.agent.process(task.input_data, task.context),
                    timeout=task.timeout_seconds
                )
                
                execution_time = (time.time() - start_time) * 1000
                
                # Record successful execution
                task.agent.record_request(True, execution_time)
                
                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    result=result,
                    execution_time_ms=execution_time,
                    agent_id=task.agent.agent_id,
                    retry_count=task.retry_count
                )
                
            except asyncio.TimeoutError:
                execution_time = (time.time() - start_time) * 1000
                task.agent.record_request(False, execution_time)
                
                # Retry if retries available
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    return await self._execute_single_task(task, semaphore)
                
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error=f"Task timeout after {task.timeout_seconds}s",
                    execution_time_ms=execution_time,
                    agent_id=task.agent.agent_id,
                    retry_count=task.retry_count
                )
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                task.agent.record_request(False, execution_time)
                
                # Retry if retries available
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    return await self._execute_single_task(task, semaphore)
                
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error=str(e),
                    execution_time_ms=execution_time,
                    agent_id=task.agent.agent_id,
                    retry_count=task.retry_count
                )
    
    async def _execute_until_first_success(
        self, 
        coroutines: List, 
        tasks: List[ParallelTask]
    ) -> Dict[str, TaskResult]:
        """Execute tasks until the first one succeeds."""
        results = {}
        pending = {asyncio.create_task(coro): i for i, coro in enumerate(coroutines)}
        
        try:
            while pending:
                done, pending_set = await asyncio.wait(
                    pending.keys(), 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    task_index = pending.pop(task)
                    task_id = tasks[task_index].task_id
                    
                    try:
                        result = await task
                        results[task_id] = result
                        
                        # If this task succeeded, cancel remaining tasks
                        if result.success:
                            for remaining_task in pending_set:
                                remaining_task.cancel()
                            
                            # Add cancelled task results
                            for remaining_task in pending_set:
                                remaining_index = next(
                                    i for t, i in pending.items() if t == remaining_task
                                )
                                remaining_task_id = tasks[remaining_index].task_id
                                results[remaining_task_id] = TaskResult(
                                    task_id=remaining_task_id,
                                    success=False,
                                    error="Cancelled - first success achieved",
                                    agent_id=tasks[remaining_index].agent.agent_id
                                )
                            
                            return results
                            
                    except Exception as e:
                        results[task_id] = TaskResult(
                            task_id=task_id,
                            success=False,
                            error=str(e),
                            agent_id=tasks[task_index].agent.agent_id
                        )
                
                # Update pending to be just the set of tasks
                pending = {t: pending[t] for t in pending_set if t in pending}
        
        finally:
            # Cancel any remaining tasks
            for task in pending.keys():
                task.cancel()
        
        return results
    
    def _process_gather_results(
        self, 
        task_results: List, 
        tasks: List[ParallelTask]
    ) -> Dict[str, TaskResult]:
        """Process results from asyncio.gather."""
        results = {}
        
        for i, result in enumerate(task_results):
            task = tasks[i]
            
            if isinstance(result, TaskResult):
                results[task.task_id] = result
            elif isinstance(result, Exception):
                results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error=str(result),
                    agent_id=task.agent.agent_id
                )
            else:
                # This shouldn't happen with proper task execution
                results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error="Unexpected result type",
                    agent_id=task.agent.agent_id
                )
        
        return results
    
    def _update_execution_stats(
        self, 
        results: Dict[str, TaskResult], 
        total_execution_time: float
    ) -> None:
        """Update execution statistics."""
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful
        
        self._execution_stats["total_tasks"] += len(results)
        self._execution_stats["successful_tasks"] += successful
        self._execution_stats["failed_tasks"] += failed
        
        # Update average execution time
        if results:
            avg_task_time = sum(r.execution_time_ms for r in results.values()) / len(results)
            current_avg = self._execution_stats["average_execution_time"]
            total_tasks = self._execution_stats["total_tasks"]
            
            self._execution_stats["average_execution_time"] = (
                (current_avg * (total_tasks - len(results)) + avg_task_time * len(results)) 
                / total_tasks
            )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get current execution statistics."""
        return self._execution_stats.copy()
    
    def get_active_tasks(self) -> List[str]:
        """Get list of currently active task IDs."""
        return [
            task_id for task_id, task in self._active_tasks.items() 
            if not task.done()
        ]
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific active task."""
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            if not task.done():
                task.cancel()
                return True
        return False
    
    async def cancel_all_tasks(self) -> int:
        """Cancel all active tasks. Returns number of tasks cancelled."""
        cancelled_count = 0
        for task in self._active_tasks.values():
            if not task.done():
                task.cancel()
                cancelled_count += 1
        return cancelled_count
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._execution_stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0.0,
            "peak_concurrent_tasks": 0
        } 