#  Copyright (c) Tabnine LTD  2023.
#  All Rights Reserved.

import os
from typing import Dict
import numpy as np

from transformers import AutoTokenizer
from configurations import (
    get_model_conf,
)
import streaming_request
from exec import execute


class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        self.gen_name = args["model_name"]
        self.fallback_model = None
        current_path: str = (
            os.path.dirname(args["model_repository"])
            if os.environ.get("NVIDIA_TRITON_SERVER_VERSION", "1") == "22.07"
            else os.path.join(args["model_repository"], args["model_version"])
        )
        print(f'model_name: {args["model_name"]}')
        print(f'model_repository: {args["model_repository"]}')
        print(f'model_version: {args["model_version"]}')
        print('model_instance_name: ', args["model_instance_name"])
        print(f'args: {args}')
        print(f'current_path: {current_path}')
        self.token_handler = None
        from tok_trie import TokenHandler
        self.model_instance_name = args["model_instance_name"]
        self.token_handler = TokenHandler
        self.config_path = os.path.join(
            current_path, os.environ.get("MODEL_CONFIG_PATH", ".")
        )
        print(f'config_path: {self.config_path}')
        self.random_type = np.uint64
        self.token_chars_heuristic = 10
        self.models = {}
        model_confs_dir = os.path.join(self.config_path, "confs")
        if os.path.exists(model_confs_dir):
            for model_conf_name in os.listdir(model_confs_dir):
                model_conf_path = os.path.join(model_confs_dir, model_conf_name)
                if os.path.isdir(model_conf_path):
                    tokenizer_path = os.path.join(model_conf_path, "tokenizer")
                    if not os.path.isdir(tokenizer_path):
                        continue
                    else:
                        self.models[model_conf_name] = {
                            "tokenizer": AutoTokenizer.from_pretrained(
                                os.path.join(model_conf_path, "tokenizer"),
                                trust_remote_code=True,
                            )
                        }
                        self.models[model_conf_name]["conf"] = get_model_conf(
                            model_conf_path,
                            self.models[model_conf_name]["tokenizer"],
                            self.token_handler,
                        )
        else:
            model_name = (
                "fastertransformer"
                if args["model_name"] == "ft_generate"
                else args["model_name"].replace("_generate", "_model")
            )
            self.fallback_model = model_name
            self.models[model_name] = {}
            if os.path.exists(os.path.join(self.config_path, "tokenizer")):
                self.models[model_name]["tokenizer"] = AutoTokenizer.from_pretrained(
                    os.path.join(self.config_path, "tokenizer"), trust_remote_code=True
                )
                self.models[model_name]["conf"] = get_model_conf(
                    self.config_path,
                    self.models[model_name]["tokenizer"],
                    self.token_handler,
                )
            else:
                raise FileNotFoundError(os.path.join(self.config_path, "tokenizer"))



    async def execute(self, requests):
        return await execute(self, requests)

    def finalize(self):
        print("Model finalizing")
        streaming_request.close_client()
