from polyfactory.factories import pydantic_factory

from speechkit.domain.dto import task


class TaskFactory(pydantic_factory.ModelFactory[task.Task]):
    pass
