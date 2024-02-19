#  Copyright (c) Tabnine LTD  2023.
#  All Rights Reserved.

# cython: language_level=3

import numpy as np
from last_empty_lines_cleanup_processor import remove_last_empty_lines
import languages
from line_utils import indent_size_of


class BaseProcessor:
    def __init__(self, tokenizer, language, input_len, unstable_text, prompt, suffix):
        self.input_len = input_len
        self.language = language
        self.tokenizer = tokenizer
        self.last_tokens = None
        self.original_unstable_text = unstable_text
        self.prompt = prompt
        self._clear()
        self.cursor_line_indent = self._calc_cursor_line_indent()
        self.suffix = suffix

    def _process_response(self, response, is_generating_done):
        print('response.output[0]' ,response.output[0])
        new_tokens = response.output[0][
            self.input_len : self.input_len + response.generated_length[0]
        ]
        print('new_tokens', new_tokens)
        # if new tokens continues last_tokens
        if self.last_tokens is not None and np.all(
            new_tokens[: len(self.last_tokens)] == self.last_tokens
        ):
            additional_text = self.tokenizer.decode(
                new_tokens[len(self.last_tokens) :], skip_special_tokens=True
            )
            #print('additional_text', additional_text)
            complete_with_line_count = self._append_text(
                additional_text, is_generating_done
            )
            print('complete_with_line_count', complete_with_line_count)
        else:
            new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            print('new_text', new_text)
            self._clear()
            complete_with_line_count = self._append_text(new_text, is_generating_done)
            print('complete_with_line_count', complete_with_line_count)

        self.last_tokens = new_tokens

        return complete_with_line_count

    def _append_text(self, text, is_generating_done):
        text = self._handle_unstable(text)
        lines = text.split("\n")
        lines[0] = self.current_line + lines[0]
        appended_lines = lines[:-1]

        if is_generating_done:
            appended_lines.append(lines[-1])

        for line in appended_lines:
            complete_with_line_count = self._append_line(line)
            if complete_with_line_count is not None:
                return complete_with_line_count

        self.current_line = lines[-1]

        return None

    def _append_line(self, line):
        for i, c in enumerate(line):
            prev_pair = self.statement_tester.active_pair()
            self.statement_tester.feed(c)
            curr_pair = self.statement_tester.active_pair()

            # If we had an "early stop" pair, and now the pair is closed, we need to check for a complete response now
            #
            # Python Example:
            # if n < 1:
            #   return 1
            # return fib(n-1) + fib(n-2)
            #
            # We want to stop as soon as the pair is closed, and not wait for the end of the line
            if (
                prev_pair is not None
                and prev_pair[0].early_stop()
                and curr_pair is None
            ):
                result = self.complete_response()
                self.lines.append(LineInfo(line[:i], False, False))
                if result is not None:
                    return result
                self.lines.pop()

        self.statement_tester.feed("\n")

        self.lines.append(
            LineInfo(
                line, self.statement_tester.opens_scope(), self.is_open_statement()
            )
        )

        complete_response_line_index = self.complete_response()
        repetitive_lines_count = self._repetitive_lines_count()
        if repetitive_lines_count is not None:
            repetitive_sequence_line_index = len(self.lines) - repetitive_lines_count
            if complete_response_line_index is not None:
                return min(repetitive_sequence_line_index, complete_response_line_index)
            return repetitive_sequence_line_index

        return complete_response_line_index

    def _handle_unstable(self, text):
        if len(self.unstable_text) == 0:
            return text
        print('self.unstable_text', self.unstable_text)
        print('text', text)
        if len(self.unstable_text) > len(text):
            assert self.unstable_text.startswith(text)
            self.unstable_text = self.unstable_text[len(text) :]
            return ""
        print('text', text)
        print('self.unstable_text', self.unstable_text)
        assert text.startswith(self.unstable_text)
        text = text[len(self.unstable_text) :]
        self.unstable_text = ""
        return text

    def _clear(self):
        self.current_line = ""
        self.lines = []
        self.unstable_text = self.original_unstable_text
        self.statement_tester = self.language.open_statement_tester(self.prompt)

    # returns None if should not stop, or the number of lines to return in the response
    def complete_response(self):
        raise NotImplementedError

    def is_open_statement(self):
        return self.statement_tester.is_open()

    def backtrack(self, response, line_count):
        text = "\n".join(map(lambda line_info: line_info.line, self.lines[:line_count]))
        line_tokens = self.tokenizer.encode(self.original_unstable_text + text + "\n")
        return response.slice(self.input_len + len(line_tokens))

    def process_and_get_final_response(self, response, is_generating_done):
        print('Non None Processor')
        print('response', response.output)
        print('is_generating_done', is_generating_done)
        complete_with_line_count = self._process_response(response, is_generating_done)
        print('complete_with_line_count', complete_with_line_count)
        if complete_with_line_count is not None:
            print('starting backtrack')
            return self.backtrack(response, complete_with_line_count)

        return None

    def _is_line_out_of_cursor_indent(self, line):
        return len(line) > 0 and indent_size_of(line) < self.cursor_line_indent

    def _calc_cursor_line_indent(self):
        last_nl_index = self.prompt.rfind("\n")
        last_line = self.prompt[last_nl_index + 1 :]
        return indent_size_of(last_line)

    def _suffix_first_line(self):
        if self.suffix:
            rest_of_suffix = self.suffix
            next_line_end_index = rest_of_suffix.find("\n")
            while next_line_end_index >= 0:
                next_line_text = rest_of_suffix[:next_line_end_index]
                if len(next_line_text.strip()) > 0:
                    return next_line_text
                rest_of_suffix = rest_of_suffix[next_line_end_index + 1 :]
                next_line_end_index = rest_of_suffix.find("\n")
            return rest_of_suffix

        return ""

    def _is_suffix_start_with_closing_bracket(self, suffix_first_line):
        return any(
            suffix_first_line.lstrip().startswith(closing_bracket)
            for closing_bracket in self.language.closing_brackets()
        )

    def _repetitive_lines_count(self):
        # if we got 4 lines that contain the same 2 lines repetitively
        if self._is_repetitive_lines_block(2):
            # if we got 4 lines that contain the same line repetitively - go back to the first line
            if self._is_repetitive_lines_block(1):
                return 3
            return 2
        # if we got 6 lines that contain the same 3 lines repetitively
        if self._is_repetitive_lines_block(3):
            return 3

        return None

    def _is_repetitive_lines_block(self, num_of_lines):
        if len(self.lines) < num_of_lines * 2:
            return False

        for index in range(num_of_lines):
            if (
                self.lines[-(index + 1)].line
                != self.lines[-(index + 1 + num_of_lines)].line
            ):
                return False

        return True

    def run_completions_postprocess(self, completions):
        return completions

    def line_index_on_out_of_cursor_indent(self):
        last_generated_line = self.lines[-1].line
        if len(self.lines) > 1 and self._is_line_out_of_cursor_indent(
            last_generated_line
        ):
            if not self.language.match_to_suffix() or self.suffix is None:
                return len(self.lines) - 1

            suffix_first_line = self._suffix_first_line()
            suffix_first_line_indent = indent_size_of(suffix_first_line)
            last_generated_line_indent = indent_size_of(last_generated_line)

            if suffix_first_line_indent > last_generated_line_indent:
                return len(self.lines) - 1

            if (
                suffix_first_line_indent == last_generated_line_indent
                and self._is_suffix_start_with_closing_bracket(suffix_first_line)
            ):
                return len(self.lines) - 1

            return len(self.lines)

        return None

    def process_max_tokens_reached_response(self, response):
        print('response', response)
        print('self.lines', self.lines)
        for index in range(1, len(self.lines) + 1):
            if (
                not self.lines[-index].is_inside_open_statement
                and self.lines[-index].line.strip() != ""
            ):
                return self.backtrack(response, len(self.lines) - index + 1)

        for index in range(len(self.lines)):
            if self.lines[index].opens_scope:
                return self.backtrack(response, index + 1)

        return None


class AlwaysIncompleteProcessor:
    def process_and_get_final_response(self, _response, _is_generating_done):
        print('Always Incomplete Processor')
        return None

    def process_max_tokens_reached_response(self, _response):
        return None

    def run_completions_postprocess(self, completions):
        return completions


class LineInfo:
    def __init__(self, line, opens_scope, is_inside_open_statement):
        self.line = line
        self.opens_scope = opens_scope
        self.is_inside_open_statement = is_inside_open_statement


class StoppingConditionProcessor(BaseProcessor):
    def complete_response(self):
        pass

    def run_completions_postprocess(self, completions):
        return list(map(remove_last_empty_lines, completions))


class StatementProcessor(StoppingConditionProcessor):
    def __init__(self, tokenizer, language, input_len, unstable_text, prompt, suffix):
        super().__init__(tokenizer, language, input_len, unstable_text, prompt, suffix)
        self._not_allowed_line_endings = (
            list(map(lambda x: " " + x, self.language.binary_operators()))
            + self.language.not_allowed_endings_for_statement()
        )

    def complete_response(self):
        if len(self.lines) > 5:
            for index in range(5):
                if self.lines[index].opens_scope:
                    return index + 1

            # if no line opens a scope than return the first 5 lines generated
            return 5
        index = self.line_index_on_out_of_cursor_indent()
        if index:
            return index

        last_line_trimmed = self.lines[-1].line.strip()
        if len(self.lines) == 1 and len(last_line_trimmed) == 0:
            return None

        if (
            self.is_open_statement()
            or any(
                last_line_trimmed.endswith(ending)
                for ending in self._not_allowed_line_endings
            )
            or any(
                last_line_trimmed.startswith(start)
                for start in self.language.not_allowed_starts_for_statement()
            )
        ):
            return None

        return len(self.lines)


class BlockProcessor(StoppingConditionProcessor):
    def __init__(self, tokenizer, language, input_len, unstable_text, prompt, suffix):
        super().__init__(tokenizer, language, input_len, unstable_text, prompt, suffix)

    def complete_response(self):
        index = self.line_index_on_out_of_cursor_indent()
        if index:
            return index
        return None


class RootScopeProcessor(StoppingConditionProcessor):
    def __init__(
        self,
        tokenizer,
        language,
        input_len,
        unstable_text,
        prompt,
        suffix,
        min_lines_to_generate,
        max_lines_to_generate,
    ):
        super().__init__(tokenizer, language, input_len, unstable_text, prompt, suffix)
        self.min_lines_to_generate = min_lines_to_generate
        self.max_lines_to_generate = max_lines_to_generate

    def complete_response(self):
        if len(self.lines) > self.max_lines_to_generate:
            for index in range(1, len(self.lines) + 1):
                if (
                    not self.lines[-index].is_inside_open_statement
                    and self.lines[-index].line.strip() != ""
                ):
                    return len(self.lines) - index + 1

            for index in range(self.max_lines_to_generate):
                if self.lines[index].opens_scope:
                    return index + 1

        if len(self.lines) > self.min_lines_to_generate:
            if not self.is_open_statement():
                return len(self.lines)

        last_line = self.lines[-1].line
        if self.language.is_comment_line(last_line) and indent_size_of(last_line) == 0:
            return len(self.lines) - 1

        return None


def get_processor(
    enable_stopping_condition,
    stop_nl,
    tokenizer,
    language,
    prompt_len,
    unstable_text,
    prompt,
    suffix,
):
    if enable_stopping_condition:
        lang = languages.get_language(language)
        if lang:
            if not stop_nl and lang.is_root_scope(prompt):
                return True, RootScopeProcessor(
                    tokenizer, lang, prompt_len, unstable_text, prompt, suffix, 5, 30
                )
            if stop_nl or not lang.is_inside_function(prompt):
                return True, StatementProcessor(
                    tokenizer, lang, prompt_len, unstable_text, prompt, suffix
                )
            else:
                return True, BlockProcessor(
                    tokenizer, lang, prompt_len, unstable_text, prompt, suffix
                )

    return False, AlwaysIncompleteProcessor()
