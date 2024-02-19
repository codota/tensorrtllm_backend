#  Copyright (c) Tabnine LTD  2023.
#  All Rights Reserved.


def remove_last_empty_lines(completion):
    while len(completion["text"]) > 0:
        completion_text = completion["text"]
        last_line_break_index = completion_text.rfind("\n")
        last_line_text = completion_text[last_line_break_index + 1 :]
        if len(last_line_text.strip()) > 0:
            return completion
        completion["text"] = completion_text[:last_line_break_index]

    return completion
