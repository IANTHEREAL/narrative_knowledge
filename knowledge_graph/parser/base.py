from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Union


@dataclass
class Index:
    name: str
    children: List = field(default_factory=list)


@dataclass
class Block:
    name: str
    content: str
    position: Optional[int] = 0


@dataclass
class SourceData:
    name: str
    content: str
    blocks: List[Block] = field(default_factory=list)
    indexes: Union[Index, List[Index], None] = None


class BaseParser(ABC):
    """
    Abstract base class for document parsers.
    """

    @abstractmethod
    def parse(self, path: str, **kwargs: Any) -> SourceData:
        """
        Parse the document located at the given path.

        Parameters:
        - path: Path to the document file.
        - **kwargs: Additional keyword arguments specific to the parser implementation.

        Returns: SourceData
        """
        pass
