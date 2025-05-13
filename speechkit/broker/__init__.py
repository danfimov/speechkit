from taskiq.abc.broker import AsyncBroker

from speechkit import dependencies


broker = dependencies.sync_container.get(AsyncBroker)
