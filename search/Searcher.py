import dataclasses
from typing import List

import lucene
from java.nio.file import *
from org.apache.lucene.store import *
from org.apache.lucene.index import *
from org.apache.lucene.search import *
from org.apache.lucene.analysis.standard import *
from org.apache.lucene.analysis.tokenattributes import *
from org.apache.lucene.util import *


@dataclasses.dataclass
class Source:
    title: str
    body: str

    @staticmethod
    def from_doc(doc):
        return Source(doc.get("title"), doc.get("body"))

    def __str__(self):
        return f"title: {self.title}\nbody: {self.body}"


class Searcher:
    def __init__(self, index_path: str, query_limit: int):
        self.path = Paths.get(index_path)
        self._directory = None
        self._reader = None
        self._searcher = None
        self.query_limit = query_limit
        self.body_name = "body"
        self._analyzer = StandardAnalyzer()
        self._phrase_query_builder = QueryBuilder(self._analyzer)
        self._bool_query_builder = BooleanQuery.Builder()

    def __enter__(self):
        if self._directory is not None:
            raise RuntimeError("Searcher already opened")
        self._directory = FSDirectory.open(self.path)
        self._reader = DirectoryReader.open(self._directory)
        self._searcher = IndexSearcher(self._reader)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._directory is None:
            raise RuntimeError("Searcher already closed")
        self._directory.close()
        self._reader.close()
        self._directory = None

        return False

    def open(self):
        return self.__enter__()

    def close(self):
        return self.__exit__(None, None, None)

    def split_text(self, text: str) -> List[str]:
        arr = []
        stream = self._analyzer.tokenStream("body", text)
        char_attr = stream.addAttribute(CharTermAttribute.class_)

        stream.reset()
        while stream.incrementToken():
            s = char_attr.toString()
            arr.append(s)
        stream.end()
        stream.close()

        return arr

    def add_clause(self, terms: List[str]):
        query = self._phrase_query_builder.createPhraseQuery(self.body_name, " ".join(terms))
        self._bool_query_builder.add(query, BooleanClause.Occur.SHOULD)

    def search(self) -> List[Source]:
        query = self._bool_query_builder.build()
        hits = self._searcher.search(query, self.query_limit)
        docs = map(lambda hit: self._searcher.doc(hit.doc), hits.scoreDocs)
        return list(map(Source.from_doc, docs))
