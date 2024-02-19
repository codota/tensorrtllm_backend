#  Copyright (c) Tabnine LTD  2023.
#  All Rights Reserved.

# cython: language_level=3

from abc import ABC
import re
from line_utils import indent_size_of
from open_statement_tester import (
    OpenStatementTester,
    SimplePair,
    ElementTag,
    SimpleEscapingPair,
    NoSpaceBeforePair,
    IndentBasedPair,
    TypescriptGenericPair,
)


class LanguageBase:
    def open_pairs(self):
        raise NotImplementedError

    def binary_operators(self):
        raise NotImplementedError

    def container_keywords(self):
        raise NotImplementedError

    def modifier_keywords(self):
        raise NotImplementedError

    def open_statement_tester(self, prompt):
        return OpenStatementTester(self.open_pairs(), prompt)

    def comment_begin_tokens(self):
        raise NotImplementedError

    def is_root_scope(self, text):
        last_nl_index = text.rfind("\n")
        last_line = text[last_nl_index + 1 :]
        return indent_size_of(last_line) == 0

    def is_comment_line(self, line):
        stripped_line = line.lstrip()
        return any(
            stripped_line.startswith(token) for token in self.comment_begin_tokens()
        )

    def is_inside_function(self, text):
        last_nl_index = text.rfind("\n")
        last_line = text[last_nl_index + 1 :]
        last_line_indent = indent_size_of(last_line)

        if last_line_indent == 0:
            return False

        current_line_index = last_nl_index
        while current_line_index >= 0:
            prev_line_index = current_line_index
            current_line_index = text.rfind("\n", 0, prev_line_index)
            current_line = text[current_line_index + 1 : prev_line_index]
            current_line_indent = indent_size_of(current_line)

            if current_line_indent < last_line_indent and len(current_line.strip()) > 1:
                stripped_current_line = current_line.lstrip()
                if (
                    len(stripped_current_line) > 0
                    and stripped_current_line[0].isalpha()
                ):
                    return not self._line_is_a_container(stripped_current_line)

        return True

    def _line_is_a_container(self, line):
        container_matches, _ = self._match_next_word(line, self.container_keywords())
        if container_matches:
            return True

        modifier_matches, remaining_line = self._match_next_word(
            line, self.modifier_keywords()
        )
        if modifier_matches:
            return self._line_is_a_container(remaining_line)

        return False

    def _match_next_word(self, line, patterns):
        match_len = 0
        for pattern in patterns:
            if type(pattern) is re.Pattern:
                match = pattern.match(line)
                if match:
                    match_len = match.end()
            else:
                if line.startswith(pattern):
                    match_len = len(pattern)

            if match_len == len(line):
                return True, ""

            space_found = False
            while 0 < match_len < len(line) and line[match_len].isspace():
                space_found = True
                match_len += 1

            if space_found:
                return True, line[match_len:]

        return False, None

    def match_to_suffix(self):
        return False

    def closing_brackets(self):
        raise NotImplementedError

    def not_allowed_endings_for_statement(self):
        return [","]

    def not_allowed_starts_for_statement(self):
        raise NotImplementedError


class CurlyBasedLanguage(LanguageBase, ABC):
    def __init__(self):
        self._open_pairs = [
            SimplePair("(", ")", True, False),
            SimplePair("[", "]", True, False),
            SimplePair("{", "}", True, True),
            SimpleEscapingPair('"', '"', False, False),
            SimpleEscapingPair("'", "'", False, False),
            SimplePair("//", "\n", False, False),
            SimplePair("/*", "*/", False, False),
        ]

        self._binary_operators = [
            "+",
            "-",
            "*",
            "/",
            "%",
            "&&",
            "||",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
            "<<",
            ">>",
        ]

        self._closing_brackets = ["}", "]", ")"]

    def open_pairs(self):
        return self._open_pairs

    def binary_operators(self):
        return self._binary_operators

    def add_pairs(self, pairs):
        self._open_pairs.extend(pairs)

    def add_closing_brackets(self, closing_brackets):
        self._closing_brackets.extend(closing_brackets)

    def match_to_suffix(self):
        return True

    def comment_begin_tokens(self):
        return ["//", "/*"]

    def closing_brackets(self):
        return self._closing_brackets


class GenericBasedLanguage(CurlyBasedLanguage):
    def __init__(self):
        super().__init__()
        self.add_pairs([NoSpaceBeforePair("<", ">", True, False)])


class CssLanguage(CurlyBasedLanguage):
    def __init__(self):
        super().__init__()

    def container_keywords(self):
        return []

    def modifier_keywords(self):
        return []

    def not_allowed_starts_for_statement(self):
        return []


class JavaScriptLanguage(CurlyBasedLanguage):
    def __init__(self):
        super().__init__()
        self.add_pairs([SimpleEscapingPair("`", "`", False, False)])

    def container_keywords(self):
        return ["class"]

    def modifier_keywords(self):
        return []

    def not_allowed_starts_for_statement(self):
        return ["@"]


class TypeScriptLanguage(JavaScriptLanguage):
    def __init__(self):
        super().__init__()
        self.add_pairs([TypescriptGenericPair()])

    def container_keywords(self):
        return ["class", "interface", "type"]

    def modifier_keywords(self):
        return ["abstract", "export", "default"]


class JavaLanguage(GenericBasedLanguage):
    def container_keywords(self):
        return ["class", "interface", "enum"]

    def modifier_keywords(self):
        return [
            "public",
            "private",
            "protected",
            "internal",
            "abstract",
            "static",
            "final",
            "sealed",
        ]

    def not_allowed_starts_for_statement(self):
        return ["@"]


class KotlinLanguage(GenericBasedLanguage):
    def container_keywords(self):
        return ["class", "interface", "enum"]

    def modifier_keywords(self):
        return [
            "open",
            "abstract",
            "inner",
            "sealed",
            "data",
        ]

    def not_allowed_starts_for_statement(self):
        return ["@"]


class DartLanguage(GenericBasedLanguage):
    def container_keywords(self):
        return ["class", "enum", "typedef"]

    def modifier_keywords(self):
        return [
            "abstract",
            "sealed",
        ]

    def not_allowed_starts_for_statement(self):
        return ["@"]


class CppLanguage(GenericBasedLanguage):
    def container_keywords(self):
        return [
            "class",
            "namespace",
            "struct",
            "enum",
            "union",
            "public:",
            "private:",
            "protected:",
        ]

    def modifier_keywords(self):
        return [re.compile("template\\<.*\\>")]

    def not_allowed_starts_for_statement(self):
        return ["template"]


class CsharpLanguage(GenericBasedLanguage):
    def container_keywords(self):
        return [
            "class",
            "interface",
            "enum",
            "namespace",
        ]

    def modifier_keywords(self):
        return [
            "public",
            "private",
            "protected",
            "abstract",
            "partial",
            "sealed",
            "static",
        ]

    def not_allowed_starts_for_statement(self):
        return ["["]

    def not_allowed_endings_for_statement(self):
        return super().not_allowed_endings_for_statement() + [")"]


class PhpLanguage(CurlyBasedLanguage):
    def container_keywords(self):
        return ["class"]

    def modifier_keywords(self):
        return []

    def not_allowed_starts_for_statement(self):
        return []


class RustLanguage(GenericBasedLanguage):
    def container_keywords(self):
        return ["struct", "trait", "enum", "mod", "impl", re.compile("impl\\<.*\\>")]

    def modifier_keywords(self):
        return [
            "pub",
            re.compile("pub\\(.*\\)"),
        ]

    def not_allowed_starts_for_statement(self):
        return ["#["]


class GoLanguage(CurlyBasedLanguage):
    def container_keywords(self):
        return ["type"]

    def modifier_keywords(self):
        return []

    def not_allowed_starts_for_statement(self):
        return ["// @"]


class PythonLanguage(LanguageBase):
    def __init__(self):
        self._open_pairs = [
            SimplePair("(", ")", True, False),
            SimplePair("[", "]", True, False),
            SimplePair("{", "}", True, False),
            SimpleEscapingPair('"', '"', False, False),
            SimpleEscapingPair("'", "'", False, False),
            SimplePair("#", "\n", False, False),
            IndentBasedPair(":\n", True, True),
        ]

        self._binary_operators = [
            "+",
            "-",
            "*",
            "/",
            "%",
            "&&",
            "||",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
            "<<",
            ">>",
        ]

    def open_pairs(self):
        return self._open_pairs

    def binary_operators(self):
        return self._binary_operators

    def container_keywords(self):
        return ["class"]

    def modifier_keywords(self):
        return []

    def not_allowed_starts_for_statement(self):
        return ["@"]

    def comment_begin_tokens(self):
        return ["#"]


class HtmlLanguage(JavaScriptLanguage):
    def __init__(self):
        super().__init__()
        self.add_pairs(
            [
                ElementTag(),
                SimplePair("<!", ">", False, False),
            ]
        )
        self.add_closing_brackets(["</", "/>"])

    def not_allowed_endings_for_statement(self):
        return super().not_allowed_endings_for_statement() + ['"']


class JsxLanguage(HtmlLanguage):
    def __init__(self):
        super().__init__()
        self.add_pairs(
            [
                SimplePair("<>", "</>", True, True),
            ]
        )


class TsxLanguage(JsxLanguage):
    def __init__(self):
        super().__init__()
        self.add_pairs([TypescriptGenericPair()])


class VueLanguage(TsxLanguage):
    def __init__(self):
        super().__init__()


def get_language(language_name):
    language_name = language_name.lower()
    if language_name == "javascript" or language_name == "js":
        return JavaScriptLanguage()
    elif language_name == "typescript":
        return TypeScriptLanguage()
    elif language_name == "java":
        return JavaLanguage()
    elif language_name == "rust":
        return RustLanguage()
    elif language_name == "python":
        return PythonLanguage()
    elif language_name == "html":
        return HtmlLanguage()
    elif language_name == "jsx":
        return JsxLanguage()
    elif language_name == "tsx":
        return TsxLanguage()
    elif language_name == "vue":
        return VueLanguage()
    elif language_name == "c/c++":
        return CppLanguage()
    elif language_name == "go":
        return GoLanguage()
    elif language_name == "kotlin":
        return KotlinLanguage()
    elif language_name == "css":
        return CssLanguage()
    elif language_name == "php":
        return PhpLanguage()
    elif language_name == "c#":
        return CsharpLanguage()
    elif language_name == "dart":
        return DartLanguage()
    else:
        return None
