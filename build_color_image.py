import requests
from bs4 import BeautifulSoup

"""
Script:
Scraps wiki page, find necessary word, colored it necessary color and add page with this word
"""

page_name = "https://en.wikipedia.org/wiki/Elvis_Presley"
word = "Elvis"  # будем тут искать по этому слову (эмбедингу/токену)
color = "red"  # цвет раскраски


def main() -> None:
    response = requests.get(page_name)
    soup = BeautifulSoup(response.content, "html.parser")

    tag = soup.new_tag("span")
    href_tag = soup.new_tag("a", href=page_name, style=f"color:{color}")
    href_tag.string = word
    tag.append(href_tag)

    for element in soup.find_all(text=True):
        if word in element and element.parent.name != "script":
            element.replace_with(str(element).replace(word, str(tag)))

    with open("coloredPage.html", "w", encoding="utf-8") as f:
        f.write(soup.prettify(formatter=None))


if __name__ == "__main__":
    main()
