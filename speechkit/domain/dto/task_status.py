import enum


class TaskStatus(enum.IntEnum):
    CREATED = 0
    PREPROCESSED = 1
    FAILED = 2
    DONE = 3
