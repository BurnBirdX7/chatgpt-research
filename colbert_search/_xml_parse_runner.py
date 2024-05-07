from __future__ import annotations

import xml.etree.ElementTree as ET
import typing as t
import re
from collections import defaultdict

EventHandler = t.Callable[[ET.Element], None]

_tag_regex = re.compile(r"{(.*)}(\w+)")


class XmlParseRunner:

    def __init__(self, filepath: str):
        self._iter = ET.iterparse(filepath, events=("start-ns", "end")).__iter__()
        self._namespace: str | None = None
        self._start_handlers: t.DefaultDict[str, t.List[EventHandler]] = defaultdict(list)
        self._end_handlers: t.DefaultDict[str, t.List[EventHandler]] = defaultdict(list)

    def start_handler(self, tag_name: str, handler: EventHandler):
        self._start_handlers[tag_name].append(handler)

    def end_handler(self, tag_name: str, handler: EventHandler):
        self._end_handlers[tag_name].append(handler)

    def collect_contents(self, tag_name: str, dest: t.Dict[str, str]):
        def _handler(elem: ET.Element):
            dest[tag_name] = elem.text

        self.end_handler(tag_name, _handler)

    def run(self):
        try:
            while True:
                self._next()
        except StopIteration:
            return

    def _tag_name(self, elem_name: str) -> str:
        if self._namespace is None:
            return elem_name
        else:
            m = _tag_regex.match(elem_name)
            if m is None:
                return elem_name

            if m.group(1) != self._namespace:
                return elem_name

            return m.group(2)

    def _next(self):
        event: str
        elem: t.Tuple[str, str] | ET.Element
        event, elem = self._iter.__next__()
        if event == "start-ns":
            if elem[0] == "":
                self._namespace = elem[1]
            return

        tag_name = self._tag_name(elem.tag)

        if event == "start" and tag_name in self._start_handlers:
            for handler in self._start_handlers[tag_name]:
                handler(elem)

        elif event == "end" and tag_name in self._end_handlers:
            for handler in self._end_handlers[tag_name]:
                handler(elem)

    def _tag(self, tag_name):
        return f"{{{self._namespace}}}{tag_name}" if self._namespace is not None else tag_name
