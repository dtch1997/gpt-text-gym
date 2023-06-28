import abc
from typing import Tuple, Optional


class BaseTask(abc.ABC):
    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def check_success(self, state) -> bool:
        raise NotImplementedError()


class AgentIsHoldingObjTask(BaseTask):
    def __init__(self, type: str, color: str):
        self.type = type
        self.color = color

    def __str__(self):
        return f"Agent is holding {self.color} {self.type}"

    def check_success(self, state) -> bool:
        raise NotImplementedError()


class AgentSeesObjTask(BaseTask):
    """
    Return True if agent sees an object in its field of view
    If position is None, object can be anywhere in the field of view
    If position is a tuple, object must be at that position
    """

    def __init__(
        self,
        num: int,
        type: str,
        color: str,
        position: Optional[Tuple[int, int]] = None,
    ):
        self.num = num
        self.type = type
        self.color = color
        self.position = position

    def __str__(self):
        if self.position is None:
            return f"Agent sees {self.num} {self.color} {self.type}"
        else:
            return f"Agent sees {self.num} {self.color} {self.type} at {self.position}"

    def check_success(self, state) -> bool:
        raise NotImplementedError()


# Logical compositions


class InverseTask(BaseTask):
    def __init__(self, task: BaseTask):
        self.task = task

    def __str__(self):
        return f"NOT({self.task})"

    def check_success(self, state) -> bool:
        return not self.task.check_success(state)


class AndTask(BaseTask):
    def __init__(self, *tasks: Tuple[BaseTask]):
        self.tasks = tasks

    def __str__(self):
        return f"AND({', '.join([str(task) for task in self.tasks])})"

    def check_success(self, state) -> bool:
        return all(task.check_success(state) for task in self.tasks)


class OrTask(BaseTask):
    def __init__(self, *tasks: Tuple[BaseTask]):
        self.tasks = tasks

    def __str__(self):
        return f"OR({', '.join([str(task) for task in self.tasks])})"

    def check_success(self, state) -> bool:
        return any(task.check_success(state) for task in self.tasks)
