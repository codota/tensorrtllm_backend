#  Copyright (c) Tabnine LTD  2023.
#  All Rights Reserved.

# cython: language_level=3


def indent_size_of(line):
    return len(line) - len(line.lstrip())
