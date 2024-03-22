from __future__ import annotations

import json
import http.client
import urllib.parse as urlparse
from typing import Dict, List

from src.pipeline import BaseNode
from src.pipeline.data_descriptors import DictDescriptor
from src.config import ColbertServerConfig


class QueryColbertServer(BaseNode):
    """
    Node accepts request and queries possible sources from colbert index

    Returns dictionary (source_url -> source_text)

    Prerequisite: colbert server must be running
    """

    def __init__(self, name: str, config: ColbertServerConfig):
        super().__init__(name, [str], DictDescriptor())
        self.server_config: ColbertServerConfig = config

    def request(self, conn: http.client.HTTPConnection, text: str) -> List[Dict[str, str]]:
        url = self.server_config.get_api_url() + '?query=' + urlparse.quote_plus(text)
        conn.request("GET", url)
        response = conn.getresponse()
        content = response.read().decode('utf-8')
        return json.loads(content)['topk']

    def process(self, text: str) -> dict:
        conn = http.client.HTTPConnection(self.server_config.ip_address, self.server_config.port)

        words = text.split(' ')
        if len(words) <= 100:
            lst_of_dicts = self.request(conn, text)
            conn.close()
            return {
                dic['source_url']: dic['text']
                for dic in lst_of_dicts
            }

        # TODO: Track the score and leave only top-k results
        accumulated_sources: Dict[str, str] = {}
        for start_pos in range(0, len(words), 50):
            req_lst: List[Dict[str, str]] = self.request(conn, ' '.join(words[start_pos:start_pos+100]))
            accumulated_sources.update({
                dic['source_url']: dic['text']
                for dic in req_lst
            })
        conn.close()

        return accumulated_sources

    def prerequisite_check(self) -> str | None:
        conn = http.client.HTTPConnection(self.server_config.ip_address, self.server_config.port)
        try:
            conn.request("GET", self.server_config.api_ping_path)
            text = conn.getresponse().read().decode('utf-8')
            if text != "colbert-pong":
                return f"Request was successful but resulted in unexpected response: {text}"

        except Exception as e:
            return f"Couldn't make a request to colbert server, cfg: {self.server_config}"

        finally:
            conn.close()

        return None
