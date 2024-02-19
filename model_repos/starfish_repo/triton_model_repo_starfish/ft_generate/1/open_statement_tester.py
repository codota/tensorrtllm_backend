#  Copyright (c) Tabnine LTD  2023.
#  All Rights Reserved.

# cython: language_level=3

from line_utils import indent_size_of


class OpenStatementTester:
    def __init__(self, pairs, prompt):
        self.prompt = prompt
        self.pairs = pairs
        self.pair_stack = []
        self.text = ""
        self.current_line = ""

    def active_pair(self):
        return self.pair_stack[-1] if len(self.pair_stack) > 0 else None

    def opens_scope(self):
        active_pair = self.active_pair()
        if active_pair is None:
            return False

        # pylint: disable=unpacking-non-sequence
        active_pair, active_state = active_pair
        return active_pair.opens_scope(active_state)

    def feed(self, char):
        self.text += char
        self.current_line += char

        if char == "\n":
            self.current_line = ""

        active_pair = self.active_pair()

        if active_pair:
            # pylint: disable=unpacking-non-sequence
            active_pair, active_state = active_pair

            # if the current generated text ends with the current pair's end - remove it
            if active_pair.matches_end(self.text, active_state):
                self.pair_stack.pop()
                return
            # if the current pair not allowed to contain other pairs inside it - stop looking for such
            if not active_pair.allow_nested():
                return

        for pair in self.pairs:
            # if there is a state such as line indentation for later calculation of closing
            start_state = pair.matches_start(self.text, self.prompt, self.current_line)
            # if the text starts with some pair's start - add it
            if start_state:
                self.pair_stack.append((pair, start_state))
                break

    def is_open(self):
        return self.active_pair() is not None


class BasePair:
    def early_stop(self):
        return False


class SimplePair(BasePair):
    def __init__(self, start, end, allow_nested, opens_scope):
        self._allow_nested = allow_nested
        self._start = start
        self._end = end
        self._opens_scope = opens_scope

    def matches_start(self, text, _prompt, _current_line):
        return text.endswith(self._start)

    def matches_end(self, text, _state):
        return text.endswith(self._end)

    def allow_nested(self):
        return self._allow_nested

    def opens_scope(self, _state):
        return self._opens_scope


class ElementTag(BasePair):
    def __init__(self):
        self._self_closing_tags = [
            "<area",
            "<base",
            "<br",
            "<col",
            "<embed",
            "<hr",
            "<img",
            "<input",
            "<link",
            "<meta",
            "<param",
            "<source",
            "<track",
            "<wbr",
        ]

    def matches_start(self, text, _prompt, current_line):
        last_line_text_trimmed = current_line.lstrip()

        # check match to '<' followed by an alphabet letter
        # also, the current line must start with '<' or '>'
        if (
            (
                text[: len(text) - 1].endswith(" <")
                or last_line_text_trimmed.startswith("<")
                or last_line_text_trimmed.startswith(">")
            )
            and len(last_line_text_trimmed) >= 2
            and last_line_text_trimmed[-2] == "<"
            and last_line_text_trimmed[-1].isalpha()
        ):
            return {"has_met_middle_symbol": False, "has_met_open_scope_symbol": False}

    def matches_end(self, text, state):
        if self._matches_open_scope_symbol(text):
            self._set_met_open_scope_symbol_true(state)
        if self._matches_middle(text):
            self._set_met_middle_true(state)
            return
        return state["has_met_middle_symbol"] and text.endswith(">")

    def _matches_middle(self, text):
        return any(
            text.endswith(middle_symbol)
            for middle_symbol in ["/"] + self._self_closing_tags
        )

    def _set_met_middle_true(self, state):
        state["has_met_middle_symbol"] = True

    def _matches_open_scope_symbol(self, text):
        return text.endswith(">")

    def _set_met_open_scope_symbol_true(self, state):
        state["has_met_open_scope_symbol"] = True

    def allow_nested(self):
        return True

    def opens_scope(self, state):
        return state["has_met_open_scope_symbol"]


class TypescriptGenericPair(BasePair):
    def matches_start(self, text, _prompt, current_line):
        last_line_text_trimmed = current_line.lstrip()
        return (
            not last_line_text_trimmed.startswith(">")
            and not last_line_text_trimmed.startswith("<")
            and not text.endswith(" <")
            and text.endswith("<")
        )

    def matches_end(self, text, _state):
        return text.endswith(">")

    def allow_nested(self):
        return False

    def opens_scope(self, _state):
        return False


class NoSpaceBeforePair(SimplePair):
    def matches_start(self, text, prompt, current_line):
        return super().matches_start(text, prompt, current_line) and not text.endswith(
            " " + self._start
        )


class SimpleEscapingPair(BasePair):
    def __init__(self, start, end, allow_nested, opens_scope, escape_char="\\"):
        self._allow_nested = allow_nested
        self._start = start
        self._end = end
        self._opens_scope = opens_scope
        self._escaped_start = escape_char + start
        self._escaped_end = escape_char + end

    def matches_start(self, text, _prompt, _current_line):
        return text.endswith(self._start) and not text.endswith(self._escaped_start)

    def matches_end(self, text, _state):
        return text.endswith(self._end) and not text.endswith(self._escaped_end)

    def allow_nested(self):
        return self._allow_nested

    def opens_scope(self, _state):
        return self._opens_scope

    def escape_start(self):
        return self._escaped_start

    def escape_end(self):
        return self._escaped_end


# a pair that ends when the last line indent is equal or lower than the indent of the line where the pair started
# Useful for the case of python - the pair will start with `:\n` and end when the indent exits the scope
class IndentBasedPair(BasePair):
    def __init__(self, start, allow_nested, opens_scope):
        self._start = start
        self._allow_nested = allow_nested
        self._opens_scope = opens_scope

    def matches_start(self, text, prompt, _current_line):
        if text.endswith(self._start):
            return [self._get_indent_of(text[: -len(self._start)], prompt)]

    def _get_indent_of(self, text, prompt):
        last_nl_index = text.rfind("\n")
        if last_nl_index > -1:
            last_line = text[last_nl_index + 1 :]
        else:
            last_nl_index = prompt.rfind("\n")
            if last_nl_index > -1:
                last_line = prompt[last_nl_index + 1 :] + text
            else:
                last_line = prompt + text

        return indent_size_of(last_line)

    def matches_end(self, text, indent_size):
        last_nl_index = text.rfind("\n")
        if last_nl_index > -1:
            last_line = text[last_nl_index + 1 :]
            if not any(map(lambda c: not c.isspace(), last_line)):
                return False
            return indent_size[0] >= indent_size_of(last_line)
        else:
            return False

    def allow_nested(self):
        return self._allow_nested

    def opens_scope(self, _state):
        return self._opens_scope

    def early_stop(self):
        return True
