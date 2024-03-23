from __future__ import annotations

import wikipediaapi  # type: ignore

banned_title_prefixes: list[str] = [
    "Category:", "File:", "See also", "References", "External links"
]

class OnlineWiki:
    wikipedia = wikipediaapi.Wikipedia(language="en",
                                       headers={"Accept-Encoding": "gzip"},
                                       user_agent="chatgpt-research-wiki-scrapper/1.1"
                                                  "(https://github.com/BurnBirdX7/chatgpt-research; "
                                                  "artemiy.lazarevx7@gmail.com)"
                                                  "Wikipedia-API/0.6.0")

    @staticmethod
    def __traverse_sections(section: wikipediaapi.WikipediaPageSection,
                            page_url: str) -> dict[str, str]:
        title = section.title

        for prefix in banned_title_prefixes:
            if title.startswith(prefix):
                return {}

        d = dict[str, str]()
        url = page_url + "#" + title.replace(" ", "_")
        d[url] = section.text

        for subsection in section.sections:
            d.update(OnlineWiki.__traverse_sections(subsection, page_url))

        return d

    @staticmethod
    def get_sections(page_title: str | list = "Elvis_Presley") -> dict[str, str]:
        """
        :param page_title: Title of the English Wikipedia article
        :return: Dictionary <Section's URL> -> <Section's Text Content>
        """
        if isinstance(page_title, list):
            return OnlineWiki.__get_sections_from_multiple_pages(page_title)

        return OnlineWiki.__get_sections_from_page(page_title)

    @staticmethod
    def __get_sections_from_page(page_title) -> dict[str, str]:
        target_page = OnlineWiki.wikipedia.page(page_title)
        if not target_page.exists():
            return {}

        url = target_page.canonicalurl
        d = dict[str, str]()
        d[url] = target_page.summary

        for section in target_page.sections:
            d.update(OnlineWiki.__traverse_sections(section, url))

        return d

    @staticmethod
    def __get_sections_from_multiple_pages(page_titles: list[str]) -> dict[str, str]:
        d = dict[str, str]()
        for page_title in page_titles:
            d |= OnlineWiki.__get_sections_from_page(page_title)
        return d

