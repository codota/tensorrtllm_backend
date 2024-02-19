#  Copyright (c) Tabnine LTD  2023.
#  All Rights Reserved.

# cython: language_level=3

import pygtrie
import numpy as np


class TokenHandler(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.trie = pygtrie.CharTrie()
        vocab = self.tokenizer.get_vocab()
        non_space_tokens = []
        for k, v in vocab.items():
            token_text = self.tokenizer.decode([v])
            self.trie[token_text] = v
            if not (token_text.startswith(" ") or token_text.startswith("\t")):
                non_space_tokens.append(v)
        self.non_space_tokens = np.array(non_space_tokens)
        self.space_tokens = self.trie.items(" ") if self.trie.has_subtrie(" ") else []
        single_space_tokens = filter(
            lambda token: not token[0].startswith("  ") and token[0] != " ",
            self.space_tokens,
        )
        self.single_space_token_ids = np.array(
            [token[1] for token in single_space_tokens]
        )
        self.tab_tokens = self.trie.items("\t") if self.trie.has_subtrie("\t") else []
        single_tab_tokens = filter(
            lambda token: not token[0].startswith("\t\t") and token[0] != "\t",
            self.tab_tokens,
        )
        self.single_tab_token_ids = np.array([token[1] for token in single_tab_tokens])

    def get_allowed_sequences_for_prefix(
        self, tokens : np.ndarray, allow_whitespace_after_indentation=False, full_unstable=False
    ):
        # print('tokens', tokens)
        # print('tokens_type' ,type(tokens))
        tokens = list(tokens)

        # The tokenizer is not greedy when it comes to spaces
        # For example, "    return" will be encoded as ["   ", " return"] and not ["    ", "return"]
        # even though these tokens exist
        # So we handle a special case were our unstable text ends with a space to account for that
        
        last_token = tokens[-1]
        split_tokens = False
        last_token_text = None
        last_token_char = None
        if last_token:
            last_token_text = self.tokenizer.decode(
                [last_token], skip_special_tokens=True
            )
            # print('last_token_text', last_token_text)
            last_token_char = last_token_text[-1] if len(last_token_text) > 0 else None
            # print('last_token_char', last_token_char)
            if len(last_token_text) > 1 and (
                last_token_char == " " or last_token_char == "\t"
            ):
                split_tokens = True
                new_last_token_text = last_token_text[0:-1]
                tokens = (
                    tokens[0:-1]
                    + self.tokenizer.encode(new_last_token_text)
                    + self.tokenizer.encode(last_token_char)
                )
                # print('tokens2', tokens)

        if full_unstable:
            unstable_length = len(tokens)
            # print('unstable_length', unstable_length)
        else:
            unstable_length = 0
            for i in range(len(tokens)):
                current_tokens = tokens[-i - 1 :]
                unstable_text = self.tokenizer.decode(
                    current_tokens, skip_special_tokens=True
                )

                if (
                    i > 0 and self.trie.has_key(unstable_text)
                ) or self.trie.has_subtrie(unstable_text):
                    unstable_length = i + 1
            # print('unstable_length', unstable_length)
        if unstable_length == 0:
            return unstable_length, []

        unstable_tokens = tokens[-unstable_length:]
        # print('unstable_tokens l90', unstable_tokens)
        if not allow_whitespace_after_indentation:
            last_line_is_indent = self._last_line_is_indent(tokens)

            if last_line_is_indent:
                allowed_sequences_a = self._append_tokens_to_prefix(
                    unstable_tokens, self.non_space_tokens, []
                )
                # print('allowed_sequences_a', allowed_sequences_a)

                if last_token_char == " ":
                    allowed_sequences_b = self._append_tokens_to_prefix(
                        unstable_tokens[0:-1], self.single_space_token_ids, [-1]
                    )
                    # print('allowed_sequences_b1', allowed_sequences_b)
                else:
                    allowed_sequences_b = self._append_tokens_to_prefix(
                        unstable_tokens[0:-1], self.single_tab_token_ids, [-1]
                    )
                    # print('allowed_sequences_b2', allowed_sequences_b)

                if split_tokens:
                    allowed_sequences_c = self._append_tokens_to_prefix(
                        unstable_tokens[0:-2] + [last_token],
                        self.non_space_tokens,
                        [-1],
                    )
                    # print('allowed_sequences_c', allowed_sequences_c)
                    suffix_tokens = filter(
                        lambda token: token[0] != last_token_text
                        and not token[0].startswith(last_token_text + last_token_char),
                        self.trie.items(last_token_text),
                    )
                    # print('suffix_tokens', suffix_tokens)

                    allowed_sequences_d = self._append_tokens_to_prefix(
                        unstable_tokens[0:-2],
                        np.array([token[1] for token in suffix_tokens]),
                        [-1, -1],
                    )
                    # print('allowed_sequences_d', allowed_sequences_d)
                else:
                    allowed_sequences_c = np.empty((0, allowed_sequences_a.shape[1]))
                    allowed_sequences_d = np.empty((0, allowed_sequences_a.shape[1]))
                    # print('allowed_sequences_c', allowed_sequences_c)
                    # print('allowed_sequences_d', allowed_sequences_d)

                if split_tokens:
                    unstable_length -= 1

                return unstable_length, np.concatenate(
                    (
                        allowed_sequences_a,
                        allowed_sequences_b,
                        allowed_sequences_c,
                        allowed_sequences_d,
                    )
                )

        allowed_sequences = np.empty((0, unstable_length))
        for i in range(unstable_length):
            token_str = self.tokenizer.decode(
                unstable_tokens[i:], skip_special_tokens=True
            )
            # print('token_str', token_str)
            if self.trie.has_key(token_str) or self.trie.has_subtrie(token_str):
                pre_tokens = unstable_tokens[:i]
                pre_tokens_text = self.tokenizer.decode(
                    pre_tokens, skip_special_tokens=True
                )
                if token_str == " ":
                    allowed_tokens = self.space_tokens
                elif token_str == "\t":
                    allowed_tokens = self.tab_tokens
                else:
                    allowed_tokens = [token for token in self.trie.items(token_str)]
                # print('allowed_tokens', allowed_tokens)
                current_allowed_sequences = np.append(
                    np.tile([pre_tokens], (len(allowed_tokens), 1)),
                    np.array(
                        [
                            [token[1]] + [-1] * (unstable_length - len(pre_tokens) - 1)
                            for token in allowed_tokens
                        ]
                    ),
                    axis=1,
                )
                # print('current_allowed_sequences', current_allowed_sequences)

                allowed_sequences = np.append(
                    allowed_sequences, current_allowed_sequences, axis=0
                )
                # print('allowed_sequences', allowed_sequences)

        if split_tokens:
            unstable_length -= 1

        return unstable_length, allowed_sequences.astype(np.int32)

    @staticmethod
    def _append_tokens_to_prefix(prefix, additional_tokens, postfix):
        return np.concatenate(
            (
                np.tile(prefix, (len(additional_tokens), 1)),
                additional_tokens.reshape(-1, 1),
                np.tile(postfix, (len(additional_tokens), 1)),
            ),
            axis=1,
        )

    def _last_line_is_indent(self, tokens):
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        last_line = text.rsplit("\n", 1)[-1]

        return last_line and last_line.isspace()
