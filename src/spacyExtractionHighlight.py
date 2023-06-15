import re

import spacy
from spacy.matcher import Matcher


class SpacyExtractorHighlight:
    RULES = {
        "Noun and specific dep": [
            {
                "DEP": {"IN": ["compound", "amod", "ccomp", "conj", "nmod"]},
                "OP": "{1,}",
            },
            {"POS": {"IN": ["NOUN", "X", "PROPN"]}},
        ],
        "Noun and adjective": [{"POS": {"IN": ["NOUN", "X", "PROPN"]}}, {"POS": "ADJ"}],
        "Noun": [
            {
                "POS": {"IN": ["NOUN", "X", "PROPN"]},
                "LIKE_EMAIL": False,
                "LIKE_URL": False,
            }
        ],
    }

    def __init__(self, model_name: str):
        self.nlp = spacy.load(model_name)

        self.rule_matcher = Matcher(self.nlp.vocab)
        for rule_name, rule_tags in self.RULES.items():  # register rules in matcher
            self.rule_matcher.add(rule_name, [rule_tags])

    """Spacy extractor for keywords extraction with higlighting"""

    def check_parenthesis(self, txt: str) -> bool:
        """Check if parenthesis are balanced

        Parameters:
            txt (str): text to check

        Returns:
            bool: True if parenthesis are balanced, False otherwise
        """

        open_list = ["[", "{", "("]
        close_list = ["]", "}", ")"]
        stack = []
        for i in txt:
            if i in open_list:
                stack.append(i)
            elif i in close_list:
                pos = close_list.index(i)
                if stack and (open_list[pos] == stack[-1]):
                    stack.pop()
                else:
                    return False
        if len(stack) == 0:
            return True
        return False

    def clear_filter_keywords_with_pos(self, keywords: list[str, (int, int)]):
        """Clear and filter keywords

        Parameters:
            keywords (list[str]): list of keywords

        Returns:
            list[str]: list of filtered keywords
        """

        # remove non ascii tokens and with parenthesis not balanced
        keywords = [
            (keyword.strip(), pos)
            for keyword, pos in keywords
            if keyword.isascii() and self.check_parenthesis(keyword)
        ]

        for i in range(len(keywords)):
            keyword, pos = keywords[i]
            keyword = keyword.replace("\n", " ")  # remove new lines
            keyword = re.sub(" \d* ", " ", keyword)  # remove numbers
            keyword = re.sub(
                "(\s*-\s+|\s+-\s*)", "", keyword
            )  # remove hyphens surronunded by spaces

            # if all letters of a word are upper case lower it
            split_keyword = keyword.split(" ")
            for j in range(len(split_keyword)):
                if split_keyword[j].isupper():
                    split_keyword[j] = split_keyword[j].lower()
                if len(split_keyword[j]) <= 2:
                    split_keyword[j] = ""
            keyword = " ".join([word for word in split_keyword if word != ""])

            # split word by capital letters if there are no spaces
            keyword = " ".join(
                [
                    catch_group[0]
                    for catch_group in re.findall(
                        "(([\da-z]+|[A-Z\.]+)[^A-Z\s()\[\]{}]*)", keyword
                    )
                ]
            )
            keyword = keyword.lower()  # put to lower case
            keyword = re.sub("\s{2,}", " ", keyword)  # remove double spaces
            keyword = re.sub("[,\.']", "", keyword)  # remove punctuation

            if len(keyword) > 0 and keyword[-1] == "-":
                keyword = keyword[:-1]
            keywords[i] = (keyword.strip(), pos)  # remove leading and trailing spaces

        return [(keyword, pos) for keyword, pos in keywords if len(keyword) > 2]

    def remove_sub_interval(self, matches: list[tuple[int, int, int]]):
        """Remove sub intervals from a list of intervals

        Parameters:
            matches (list[tuple[int, int, int]]): list of intervals (match_id, start, end)

        Returns:
            list[tuple[int, int, int]]: list of intervals without sub intervals
        """
        keep_matches = []

        if len(matches) > 0:
            _, longest_start_id, longest_end_id = matches[0]
            id_to_add = 0
            for i, match in enumerate(matches[1:], 1):
                _, start_id, end_id = match
                if (longest_start_id <= end_id and longest_end_id >= end_id) or (
                    start_id <= longest_end_id and end_id >= longest_start_id
                ):
                    if end_id - start_id > longest_end_id - longest_start_id:
                        longest_start_id, longest_end_id = start_id, end_id
                        id_to_add = i
                else:
                    if end_id > longest_end_id:
                        keep_matches.append(matches[id_to_add])
                        longest_start_id, longest_end_id = start_id, end_id
                        id_to_add = i
        return keep_matches

    def extract(self, txt: str) -> list[str]:
        """Extract keywords from a text

        Parameters:
            txt (str): text to extract keywords from

        Returns:
            list[str]: text with keywords highlighted with in a new line as a list of lines
        """

        doc = self.nlp(txt)
        matches = self.rule_matcher(doc)  # Run matcher

        keep_matches = self.remove_sub_interval(matches)
        result_with_pos = [
            (doc[start:end].lemma_, (start, end)) for _, start, end in keep_matches
        ]
        result_with_pos = self.clear_filter_keywords_with_pos(result_with_pos)

        result = []

        prev_end = 0
        keywords_inline = []
        space_between_keywords = []
        line = ""
        for word, (start, end) in result_with_pos:
            if "\n" in doc[prev_end:start].text_with_ws:
                split_txt = doc[prev_end:start].text_with_ws.split("\n")
                line += split_txt[0]
                result.append(line)
                line = ""
                if len(keywords_inline) > 0:
                    line = " " * space_between_keywords[0]
                    i = 1
                    for w in keywords_inline:
                        line += w
                        nb_space = space_between_keywords[i] - len(w)
                        if i + 1 < len(space_between_keywords):
                            nb_space += space_between_keywords[i + 1]
                        line += " " * nb_space
                        i += 2
                    result.append(line)
                    line = ""
                keywords_inline = []
                space_between_keywords = []
                line += split_txt[1]
                space_between_keywords.append(len(split_txt[1]))
            else:
                line += doc[prev_end:start].text_with_ws
                space_between_keywords.append(len(doc[prev_end:start].text_with_ws))
            keywords_inline.append(word)
            line += doc[start:end].text_with_ws
            space_between_keywords.append(len(doc[start:end].text_with_ws))
            prev_end = end
        line += doc[prev_end:].text_with_ws
        result.extend(line.split("\n"))

        return result
