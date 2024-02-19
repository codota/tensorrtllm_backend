#  Copyright (c) Tabnine LTD  2023.
#  All Rights Reserved.

# cython: language_level=3


def remove_last_line_break(completion):
    completion_text = completion["text"]
    last_line_break_index = completion_text.rfind("\n")
    if last_line_break_index >= 0:
        completion["text"] = completion_text[:last_line_break_index]

    token_scores = completion["token_scores"]
    last_line_break_token_index = get_last_line_break_token_index(token_scores)
    if last_line_break_token_index >= 0:
        token = token_scores[last_line_break_token_index]
        token_text = token["text"]
        token_updated_text = token_text[: token_text.rfind("\n")]
        if len(token_updated_text) == 0:
            completion["token_scores"] = token_scores[:last_line_break_token_index]
        else:
            token["text"] = token_updated_text
            completion["token_scores"] = token_scores[: last_line_break_token_index + 1]

    return completion


def get_last_line_break_token_index(token_scores):
    for i in range(1, len(token_scores) + 1):
        if "\n" in token_scores[-i]["text"]:
            return len(token_scores) - i
    return -1
