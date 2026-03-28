from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from schemas.models import Document


class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[Document]:
        pass
