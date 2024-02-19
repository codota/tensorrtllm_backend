#  Copyright (c) Tabnine LTD  2023.
#  All Rights Reserved.

# cython: language_level=3

import configparser
import os
from typing import Optional
import numpy as np


def get_model_conf(model_conf_path: str, tokenizer, token_handler=None) -> dict:
    conf_dict = {}
    nl_token = [
        k for k, v in tokenizer.get_vocab().items() if v == tokenizer.encode("\n")[0]
    ][0]
    conf_dict["newline_tokens"] = [
        v for k, v in tokenizer.get_vocab().items() if nl_token in k
    ]
    tokens_trie = token_handler(tokenizer)
    conf_dict["tokens_trie"] = tokens_trie
    config_file = os.path.join(model_conf_path, "model_config.ini")
    conf_dict.update(load_config_file(config_file))
    conf_dict["mid_token_id"] = None
    conf_dict["suf_token_id"] = None
    conf_dict["pre_token_id"] = None
    conf_dict["lang_token_ids"] = {}
    conf_dict["fim_supported"] = (
        conf_dict.get("fim_suf_ratio", 0)
        and "additional_special_tokens" in tokenizer.special_tokens_map
        and conf_dict.get("suf_token", None)
        in tokenizer.special_tokens_map["additional_special_tokens"]
    )
    if conf_dict.get("fim_supported", False):
        conf_dict["mid_token_id"] = tokenizer.encode(conf_dict["mid_token"])
        conf_dict["suf_token_id"] = tokenizer.encode(conf_dict["suf_token"])
        conf_dict["pre_token_id"] = tokenizer.encode(conf_dict["pre_token"])

    conf_dict["hint_supported"] = (
        "additional_special_tokens" in tokenizer.special_tokens_map
        and conf_dict.get("soh_token", None)
        in tokenizer.special_tokens_map["additional_special_tokens"]
        and conf_dict.get("sos_token", None)
        in tokenizer.special_tokens_map["additional_special_tokens"]
        and conf_dict.get("eoh_token", None)
        in tokenizer.special_tokens_map["additional_special_tokens"]
    )

    for lang in conf_dict["lang_tokens"]:
        lang_token_id = tokenizer.encode(conf_dict["lang_tokens"][lang])
        if len(lang_token_id) == 1:
            conf_dict["lang_token_ids"][lang] = [lang_token_id]

    return conf_dict


def load_config_file(config_file_path: Optional[str] = None) -> dict:
    config = configparser.RawConfigParser()
    conf_dict = {}
    if os.path.exists(config_file_path):
        config.read(config_file_path)

    conf_dict["max_prompt_tokens"] = config.getint(
        "MODEL", "MAX_PROMPT_TOKENS", fallback=300
    )
    conf_dict["default_beams"] = config.getint("MODEL", "DEFAULT_BEAMS", fallback=2)
    conf_dict["max_generate_tokens"] = config.getint(
        "MODEL", "MAX_GENERATE_TOKENS", fallback=300
    )
    conf_dict["default_generate_tokens"] = config.getint(
        "MODEL", "DEFAULT_GENERATE_TOKENS", fallback=60
    )
    conf_dict["default_generate_tokens_statement"] = int(
        config.getint(
            "MODEL",
            "DEFAULT_GENERATE_TOKENS_STATEMENT",
            fallback=100,
        )
    )
    conf_dict["default_generate_tokens_block"] = int(
        config.getint(
            "MODEL",
            "DEFAULT_GENERATE_TOKENS_BLOCK",
            fallback=200,
        )
    )
    conf_dict["max_generate_tokens_in_stop_nl"] = config.getint(
        "MODEL", "MAX_GENERATE_TOKENS_IN_STOP_NL", fallback=20
    )
    conf_dict["model_max_context"] = config.getint(
        "MODEL", "MODEL_MAX_CONTEXT", fallback=2048
    )
    conf_dict["default_num_return_sequences"] = config.getint(
        "MODEL", "DEFAULT_RETURN_SEQUENCES", fallback=1
    )
    conf_dict["is_spm"] = config.getboolean("MODEL", "IS_SPM", fallback=False)
    conf_dict["is_hard_strip"] = config.getboolean(
        "MODEL", "IS_HARD_STRIP", fallback=False
    )
    conf_dict["is_streaming"] = config.getboolean(
        "MODEL", "IS_STREAMING", fallback=True
    )
    conf_dict["default_temperature"] = config.getfloat(
        "MODEL", "DEFAULT_TEMPERATURE", fallback=1.0
    )
    conf_dict["default_top_p"] = config.getfloat("MODEL", "DEFAULT_TOP_P", fallback=0.8)
    conf_dict["default_top_k"] = config.getfloat("MODEL", "DEFAULT_TOP_K", fallback=0.0)
    conf_dict["repetition_penalty"] = config.getfloat(
        "MODEL",
        "DEFAULT_REPETITION_PENALTY",
        fallback=1.0,
    )
    conf_dict["fim_suf_ratio"] = config.getfloat("MODEL", "FIM_SUF_RATIO", fallback=0.0)
    conf_dict["exports_ratio"] = config.getfloat("MODEL", "EXPORTS_RATIO", fallback=0.2)
    conf_dict["imports_ratio"] = config.getfloat(
        "MODEL", "IMPORTS_RATIO", fallback=0.75
    )
    conf_dict["suf_token"] = config.get("MODEL", "SUF_TOKEN", fallback="[SUF]")
    conf_dict["mid_token"] = config.get("MODEL", "MID_TOKEN", fallback="[MID]")
    conf_dict["pre_token"] = config.get("MODEL", "PRE_TOKEN", fallback="[PRE]")
    conf_dict["soh_token"] = config.get("HINT_TOKENS", "START_OF_HINT", fallback=None)
    conf_dict["sos_token"] = config.get("HINT_TOKENS", "START_OF_SUFFIX", fallback=None)
    conf_dict["eoh_token"] = config.get("HINT_TOKENS", "END_OF_HINT", fallback=None)
    conf_dict["use_new_model_scheme"] = config.getboolean(
        "MODEL", "USE_NEW_MODEL_SCHEME", fallback=False
    )
    conf_dict["lang_tokens"] = {}
    for option, section in config.items():
        if option == "LANG_TOKENS":
            for lang in section:
                conf_dict["lang_tokens"][lang] = config.get("LANG_TOKENS", lang)
    print("conf_dict",conf_dict, flush=True)
    return conf_dict


def to_word_list_format(word_dict, tokenizer):
    flat_ids = []
    offsets = []
    for word_dict_item in word_dict:
        item_flat_ids = []
        item_offsets = []

        for word in word_dict_item:
            ids = tokenizer.encode(word)

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

            # Hack, can we do this better?
            if word == "\n\n":
                item_flat_ids += [198, 198]
                item_offsets.append(2)

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))


def normalize_indentation(text, indent_size):
    return text if indent_size is None else text.replace("\t", (" " * indent_size))


def fix_completion_indent(text, indent, prompt_suf):
    if indent:
        text = normalize_indentation(text, indent)
