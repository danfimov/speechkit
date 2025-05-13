import typing as tp

import structlog
from prometheus_client import Counter, Histogram
from taskiq.abc.middleware import TaskiqMiddleware


if tp.TYPE_CHECKING:
    from taskiq.message import TaskiqMessage
    from taskiq.result import TaskiqResult


logger = structlog.get_logger(__name__)


class PrometheusMiddleware(TaskiqMiddleware):
    """Middleware that adds prometheus metrics for workers."""

    def __init__(self) -> None:
        super().__init__()
        self.found_errors = Counter(
            'found_errors',
            'Number of found errors',
            ['task_name'],
        )
        self.received_tasks = Counter(
            'received_tasks',
            'Number of received tasks',
            ['task_name'],
        )
        self.success_tasks = Counter(
            'success_tasks',
            'Number of successfully executed tasks',
            ['task_name'],
        )
        self.saved_results = Counter(
            'saved_results',
            'Number of saved results in result backend',
            ['task_name'],
        )
        self.execution_time = Histogram(
            'execution_time',
            'Time of function execution',
            ['task_name'],
        )

    def pre_execute(
        self,
        message: 'TaskiqMessage',
    ) -> 'TaskiqMessage':
        self.received_tasks.labels(message.task_name).inc()
        return message

    def post_execute(
        self,
        message: 'TaskiqMessage',
        result: 'TaskiqResult[tp.Any]',
    ) -> None:
        if result.is_err:
            self.found_errors.labels(message.task_name).inc()
        else:
            self.success_tasks.labels(message.task_name).inc()
        self.execution_time.labels(message.task_name).observe(result.execution_time)

    def post_save(
        self,
        message: 'TaskiqMessage',
        _: 'TaskiqResult[tp.Any]',
    ) -> 'None':
        self.saved_results.labels(message.task_name).inc()
