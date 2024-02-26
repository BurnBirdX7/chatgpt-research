import wikipediaapi  # type: ignore

from typing import Dict


class Wiki:
    @staticmethod
    def traverse_sections(section: wikipediaapi.WikipediaPageSection,
                          page_url: str) -> Dict[str, str]:
        d = dict()

        # Embed title into paragraph
        text = f" {'=' * section.level} {section.title} {'=' * section.level} \n"
        text += section.text

        url = page_url + "#" + section.title.replace(" ", "_")
        d[url] = text

        for subsection in section.sections:
            d.update(Wiki.traverse_sections(subsection, page_url))
        return d

    @staticmethod
    def parse(title: str = "Elvis_Presley") -> Dict[str, str]:
        """
        :param title: Title of the English Wikipedia article
        :return: Dictionary <Section's URL> -> <Section's Text Content>
        """
        wikipedia = wikipediaapi.Wikipedia(language="en",
                                           user_agent="chatgpt-research-wiki-scrapper/1.0"
                                                      "(https://github.com/BurnBirdX7/chatgpt-research; "
                                                      "artemiy.lazarevx7@gmail.com)"
                                                      "Wikipedia-API/0.6.0")
        target_page = wikipedia.page(title)
        url = target_page.canonicalurl
        d: Dict[str, str] = dict()
        d[url] = target_page.summary

        for section in target_page.sections:
            d.update(Wiki.traverse_sections(section, url))

        return d
