from unittest import mock

import pytest
import taskiq
from polyfactory import pytest_plugin

from speechkit.domain.repository import file_system, task
from tests.unit import factories


@pytest.fixture
def file_system_repository_mock() -> mock.AsyncMock:
    return mock.AsyncMock(spec=file_system.FileSystemRepository)


@pytest.fixture
def task_repository_mock():
    return mock.AsyncMock(spec=task.AbstractTaskRepository)


@pytest.fixture
def broker_mock():
    return mock.AsyncMock(spec=taskiq.AsyncBroker)


pytest_plugin.register_fixture(factory=factories.TaskFactory, name='task_factory')
