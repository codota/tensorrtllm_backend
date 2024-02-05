#  Copyright (c) Tabnine LTD  2023.
#  All Rights Reserved.

# cython: language_level=3

import json
import time
from processors import get_processor

# noinspection PyUnresolvedReferences
import triton_python_backend_utils as pb_utils  # pylint: disable=import-error
import numpy as np
import streaming_request
import languages

MAX_ALLOWED_PREFIX_LENGTH = 5


async def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
    """
    Parse and tokenize each request
    :param requests: 1 or more requests received by Triton server.
    :return: text as input tensors
    """
    responses = []
    print('hi')
    # for loop for batch requests (disabled in our case)
    for request in requests:
        start_time = time.time()
        # step 0 - binary data typed back to string
        query = [
            t.decode("UTF-8")
            for t in pb_utils.get_input_tensor_by_name(request, "TEXT")
            .as_numpy()
            .tolist()
        ]
        print(query)
        # step 1 - parse query as json
        try:
            query_payload = json.loads(query[0])
        except (ValueError, json.JSONDecodeError):
            err_msg = f"Could not deserialize JSON payload: {query[0]}"
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(err_msg)
                )
            )
            continue

        # step 2 - handle prompt
        if "prompt" not in query_payload:
            err_msg = f"Request payload does not contain ``prompt``: {query_payload}"
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(err_msg)
                )
            )
            continue

        language = query_payload.get("language", "")
        enable_stopping_condition = query_payload.get("enable_stopping_condition", True)
        print('enable_stopping_condition: ', enable_stopping_condition, flush=True)

        # HACK until we decrease timeout in NATS service instead
        time_diff = 0
        if "time_stamp" in query_payload:
            time_diff = query_payload["time_stamp"] - time.time()
            if time_diff <= 0.35:
                raise TimeoutError(
                    f"too much time passed in queue, time_diff is {time_diff}"
                )
        model_name = None
        print("query_payload: ", query_payload)
        if "model_name" not in query_payload:
            model_name = self.fallback_model
            if not (self.fallback_model and self.fallback_model in self.models):
                print('self.fallback_model: ', self.fallback_model, flush=True)
                raise NameError("model_name not found in payload")
        if not model_name and query_payload["model_name"] not in self.models:
            raise NameError(
                f'model_name {query_payload["model_name"]} not found in models'
            )
        model_name = query_payload["model_name"] if not model_name else model_name
        tokenizer = self.models[model_name]["tokenizer"]
        confs = self.models[model_name]["conf"]
        prompt = query_payload["prompt"]
        suffix = query_payload.get("preceding_text", None)
        imports = None
        exports = None
        sim_ref = None
        sim_next_lines = None
        sim_window_size = None
        sim_score = None
        exports_rate = None
        imports_rate = None
        if "enrichment" in query_payload:
            deps = query_payload["enrichment"].get("dependencies", None)
            sim_snippets = query_payload["enrichment"].get("similar_snippets", None)
            if deps:
                imports = deps.get("imports", None)
                exports = deps.get("exports", None)
                exports_rate = deps.get("exports_rate", None)
                exports_rate = (
                    confs["exports_ratio"] if exports_rate is None else exports_rate
                )
                imports_rate = (
                    confs["imports_ratio"] if imports_rate is None else imports_rate
                )
            if (
                sim_snippets
                and "snippets" in sim_snippets
                and type(sim_snippets["snippets"]) == list
                and len(sim_snippets["snippets"]) > 0
            ):
                sim_window_size = sim_snippets.get("window_size", None)
                sim_snippets = sim_snippets["snippets"][0]
                sim_ref = sim_snippets.get("snippet", None)
                sim_next_lines = sim_snippets.get("next_lines", None)
                sim_score = sim_snippets.get("score", None)
        num_beams = query_payload.get("num_beams", confs["default_beams"])
        if num_beams not in [1, 2, 4, 8, 16]:
            raise ValueError(
                f"num_beams must be either 1, 2, 4, 8, 16, not {num_beams}"
            )
        is_hinting = (
            confs["hint_supported"]
            and sim_ref
            and sim_next_lines
            and sim_window_size
            and sim_window_size > 0
        )
        stop_nl = query_payload.get("stop_nl", True)
        num_return_sequences = query_payload.get(
            "num_return_sequences", confs["default_num_return_sequences"]
        )
        num_return_sequences = (
            min(num_return_sequences, num_beams)
            if num_beams > 1
            else num_return_sequences
        )
        fim_suf_ratio = query_payload.get("fim_suf_ratio", None)
        fim_suf_ratio = (
            fim_suf_ratio if fim_suf_ratio is not None else confs["fim_suf_ratio"]
        )
        is_spm = query_payload.get("is_spm", confs["is_spm"])
        use_fim = (
            suffix is not None
            and suffix != ""
            and confs["fim_supported"]
            and fim_suf_ratio
            and fim_suf_ratio > 0.0
        )
        max_prompt_tokens = query_payload.get("max_prompt_tokens", 0)
        max_prompt_tokens = max(max_prompt_tokens, confs["max_prompt_tokens"])
        lang_token = query_payload.get("lang_token", False)
        lang = language.lower()
        if lang_token:
            lang_token_id = confs["lang_token_ids"].get(lang, None)
            if lang_token_id is not None:
                max_prompt_tokens -= 1
            else:
                lang_token = False

        max_length = None
        if enable_stopping_condition:
            if languages.get_language(language):
                max_length = query_payload.get(
                    "max_length",
                    confs["default_generate_tokens_statement"]
                    if stop_nl
                    else confs["default_generate_tokens_block"],
                )
                max_length = min(max_length, confs["max_generate_tokens"])

        if max_length is None:
            max_length = query_payload.get(
                "max_length", confs["default_generate_tokens"]
            )
            if stop_nl:
                max_length = min(max_length, confs["max_generate_tokens_in_stop_nl"])
            else:
                max_length = min(max_length, confs["max_generate_tokens"])
                print(f"max_length: {max_length}", flush=True)

        max_prompt_tokens = max(
            min(confs["model_max_context"] - max_length, max_prompt_tokens), 100
        )

        max_pre_tokens = (
            max_prompt_tokens
            if exports is None
            else int(max_prompt_tokens * (1 - exports_rate))
        )
        max_pre_tokens = (
            max_pre_tokens
            if not use_fim
            else int(max_pre_tokens - (max_prompt_tokens * fim_suf_ratio))
        )

        max_chars = max_pre_tokens * self.token_chars_heuristic
        is_max_chars_prompt = len(prompt) > max_chars
        trimmed_prompt = len(prompt) - max_chars if is_max_chars_prompt else 0
        trimmed_exports = 0
        trimmed_suf = 0
        trimmed_imports = 0
        trimmed_pre = 0
        hint_size = 0
        prompt = prompt if not is_max_chars_prompt else prompt[-max_chars:]
        external_hard_strip = query_payload.get("hard_strip", True)
        is_hard_strip = confs["is_hard_strip"] and external_hard_strip
        indent_size = query_payload.get("indent", None)
        hard_strip_suf = None
        orig_prompt = prompt
        print(f"orig_prompt: {orig_prompt}", flush=True)
        if is_hard_strip:
            last_nl = prompt.rfind("\n")
            while last_nl - 1:
                if prompt[last_nl - 1] in ["\n", "\r"]:
                    last_nl -= 1
                else:
                    break
            if last_nl != -1:
                hard_strip_suf = prompt[last_nl:]
                if hard_strip_suf.isspace():
                    prompt = prompt[:last_nl]
                else:
                    is_hard_strip = False
            else:
                is_hard_strip = False
        print('prompt after hard_strip: ', prompt, flush=True)
        prompt_ids = tokenizer.encode(prompt)

        imports_max_tokens = 0
        if imports:
            imports = imports if imports[-1] == "\n" else imports + "\n"
            imports_max_tokens = int(max_pre_tokens * imports_rate)
        imports_ids = tokenizer.encode(imports) if imports else []
        len_imports_ids = len(imports_ids) if imports_ids else 0
        if imports_ids and len_imports_ids > imports_max_tokens:
            imports_ids = imports_ids[-imports_max_tokens:]
            is_nl_in_imports = np.isin(imports_ids, confs["newline_tokens"])
            first_nl_token = (
                np.argmax(is_nl_in_imports) if len(is_nl_in_imports) > 0 else -1
            )
            imports_ids = imports_ids[first_nl_token + 1 :]
            trimmed_imports += len_imports_ids - len(imports_ids)
            len_imports_ids = len(imports_ids)

        exports_ids = None
        is_max_exports_chars = None
        is_max_exports = None
        if exports and exports_rate:
            exports = exports if exports[-1] == "\n" else exports + "\n"
            exports_max_tokens = int(max_prompt_tokens * exports_rate)
            is_max_exports_chars = (
                len(exports) > exports_max_tokens * self.token_chars_heuristic
            )
            exports = exports[-(exports_max_tokens * self.token_chars_heuristic) :]
            exports_ids = tokenizer.encode(exports)
            overflow_exports = len(exports_ids) - exports_max_tokens
            is_max_exports = overflow_exports > 0
            trimmed_exports = overflow_exports if is_max_exports else 0
            exports_ids = exports_ids[-exports_max_tokens:]
            if not is_max_exports:
                max_pre_tokens = max_pre_tokens + exports_max_tokens - len(exports_ids)

        if use_fim:
            suf_max_tokens = int(max_prompt_tokens * fim_suf_ratio)
            is_max_suff_chars = (
                len(suffix) > suf_max_tokens * self.token_chars_heuristic
            )
            suffix = suffix[: suf_max_tokens * self.token_chars_heuristic]
            suffix_ids = tokenizer.encode(suffix)
            overflow_suff = len(suffix_ids) - suf_max_tokens
            is_max_suff = overflow_suff > 0
            suffix_ids = suffix_ids[:suf_max_tokens]
            trimmed_suf = overflow_suff if is_max_suff else 0
            if is_max_suff_chars or is_max_suff:
                last_suff_nl_index = np.isin(
                    np.flip(suffix_ids), confs["newline_tokens"]
                )
                suf_last_nl = (
                    (len(suffix_ids) - np.argmax(last_suff_nl_index) - 1)
                    if len(last_suff_nl_index) > 0
                    else len(suffix_ids)
                )
                suffix_ids = suffix_ids[:suf_last_nl]
                trimmed_suf += len(suffix_ids) - suf_last_nl
            else:
                max_pre_tokens = (
                    max_pre_tokens
                    + suf_max_tokens
                    - len(suffix_ids)
                    - 3  # 3 special tokens
                )

        max_pre_tokens -= len_imports_ids
        is_max_prompt = len(prompt_ids) > max_pre_tokens
        trimmed = (
            prompt_ids[: (len(prompt_ids) - max_pre_tokens)] if is_max_prompt else None
        )
        if trimmed is not None:
            trimmed_pre += len(trimmed)
            trimmed_prompt += len(tokenizer.decode(trimmed, skip_special_tokens=True))
        prompt_ids = prompt_ids[-max_pre_tokens:]
        if is_max_prompt or is_max_chars_prompt:
            is_nl_in_prompt = np.isin(prompt_ids, confs["newline_tokens"])
            first_nl_token = (
                np.argmax(is_nl_in_prompt) if len(is_nl_in_prompt) > 0 else -1
            )
            prompt_ids = prompt_ids[first_nl_token + 1 :]
            trimmed_pre += first_nl_token
        if is_hinting:
            hint_ids = (
                confs["soh_token"]
                + sim_ref
                + confs["sos_token"]
                + sim_next_lines
                + confs["eoh_token"]
            )
            hint_ids = tokenizer.encode(hint_ids)
            prompt_nl = np.where(np.isin(prompt_ids, confs["newline_tokens"]))[0]
            if len(prompt_nl) > sim_window_size:
                sim_window_size = (
                    sim_window_size + 1
                    if prompt_nl[-1] == len(prompt_ids) - 1
                    else sim_window_size
                )
                split_idx = prompt_nl[-1 * sim_window_size]
                if len(prompt_ids[split_idx:]) + len(hint_ids) < max_pre_tokens:
                    leftovers = len(prompt_ids) + len(hint_ids) - max_pre_tokens
                    prompt_ids = np.concatenate(
                        [prompt_ids[:split_idx], hint_ids, prompt_ids[split_idx:]]
                    )
                    hint_size = len(hint_ids)
                    if leftovers > 0:
                        trim_idx = prompt_nl[np.argmax(prompt_nl > leftovers)]
                        prompt_ids = prompt_ids[trim_idx:]
                        trimmed_pre += trim_idx
        cat_list = None
        if exports_ids:
            if is_max_exports or is_max_exports_chars:
                is_nl_in_exports = np.isin(exports_ids, confs["newline_tokens"])
                first_nl_token = (
                    np.argmax(is_nl_in_exports) if len(is_nl_in_exports) > 0 else -1
                )
                exports_ids = exports_ids[first_nl_token + 1 :]
                trimmed_exports += first_nl_token
            cat_list = [exports_ids, prompt_ids]
        if imports_ids:
            imports_tokens = len(imports_ids)
            cat_list = (
                [imports_ids] + cat_list if cat_list else [imports_ids, prompt_ids]
            )
        pre_tokens = len(prompt_ids)
        if cat_list:
            pre_tokens = len(prompt_ids)
            prompt_ids = np.concatenate(cat_list)
        if use_fim:
            if is_spm:
                input_start_ids = np.expand_dims(
                    np.concatenate(
                        [
                            confs["suf_token_id"],
                            suffix_ids,
                            confs["pre_token_id"],
                            prompt_ids,
                            confs["mid_token_id"],
                        ]
                    ),
                    0,
                )
            else:
                input_start_ids = np.expand_dims(
                    np.concatenate(
                        [
                            confs["pre_token_id"],
                            prompt_ids,
                            confs["suf_token_id"],
                            suffix_ids,
                            confs["mid_token_id"],
                        ]
                    ),
                    0,
                )

            prompt = (
                confs["suf_token"]
                + suffix
                + confs["pre_token"]
                + prompt
                + confs["mid_token"]
                if is_spm
                else confs["pre_token"]
                + prompt
                + confs["suf_token"]
                + suffix
                + confs["mid_token"]
            )
        else:
            input_start_ids = np.expand_dims(prompt_ids, 0)

        expand = 1 if num_beams > 1 else num_return_sequences

        if lang_token:
            input_start_ids = np.concatenate([lang_token_id, input_start_ids], axis=1)
            prompt = confs["lang_tokens"][lang] + prompt

        input_start_ids = np.repeat(input_start_ids, expand, axis=0).astype(np.int32)
        print('whole_prompt',tokenizer.decode(input_start_ids[0], skip_special_tokens=True))
        print('input_start_ids.shape', input_start_ids.shape)
        midline_flag = query_payload.get("mid_option", True)
        use_trie = (
            confs["tokens_trie"] is not None
            and prompt[-1] != "\n"
            and midline_flag
            and not use_fim
            and not is_hard_strip
        )
        allowed_sequences = None
        unstable_text = None
        if use_trie:
            potential_unstable_tokens = input_start_ids[0, -MAX_ALLOWED_PREFIX_LENGTH:]
            unstable_length, _ = confs[
                "tokens_trie"
            ].get_allowed_sequences_for_prefix(
                potential_unstable_tokens,
                language.lower() not in ["python", "jupyter notebook"],
            )
            unstable_text = tokenizer.decode(input_start_ids[0, -unstable_length:])
           

        if is_hard_strip:
            print('inside is_hard_strip')
            potential_unstable_tokens = tokenizer.encode(hard_strip_suf)
            unstable_length, allowed_sequences = confs[
                "tokens_trie"
            ].get_allowed_sequences_for_prefix(potential_unstable_tokens, False, True)
            print("allowed_sequences", allowed_sequences)
            unstable_text = hard_strip_suf
        if is_hard_strip:
            prompt_len = input_start_ids.shape[1]
        else:
            prompt_len = input_start_ids.shape[1] - unstable_length
        input_len = prompt_len * np.ones([input_start_ids.shape[0], 1]).astype(
            np.int32
        )

        output_len = np.ones_like(input_len).astype(np.int32) * max_length

        temperature = query_payload.get("temperature", confs["default_temperature"])
        if temperature <= 0.0:
            raise ValueError("Temperature must be greater than zero")

        top_p = query_payload.get("top_p", confs["default_top_p"])
        top_k = query_payload.get("top_k", confs["default_top_k"])
        repetition_penalty = query_payload.get(
            "repetition_penalty", confs["repetition_penalty"]
        )
        # diversity_penalty = query_payload.get("diversity_penalty", None)

        if num_beams > 1:
            top_p = 1.0
            top_k = 0.0
        else:
            # diversity_penalty = 1.0
            if top_p > 0 and top_p < 1:
                top_k = 0.0
            elif top_k > 0:
                top_p = 1.0

        num_logprobs = 1
        want_logprobs = True

        runtime_top_k = (top_k * np.ones([input_start_ids.shape[0], 1])).astype(
            np.int32
        )
        runtime_top_p = (top_p * np.ones([input_start_ids.shape[0], 1])).astype(
            np.float32
        )
        beam_search_diversity_rate = 0.0 * np.ones(
            [input_start_ids.shape[0], 1]
        ).astype(np.float32)

        random_seed = np.random.randint(
            0, 2**31 - 1, (input_start_ids.shape[0], 1), dtype=self.random_type
        )
        random_seed = (
            np.repeat([[42]], input_start_ids.shape[0], axis=0).astype(self.random_type)
            if num_beams >= 1
            else random_seed
        )
        temperature = temperature * np.ones([input_start_ids.shape[0], 1]).astype(
            np.float32
        )
        len_penalty = 1.0 * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        repetition_penalty = repetition_penalty * np.ones(
            [input_start_ids.shape[0], 1]
        ).astype(np.float32)
        is_return_log_probs = want_logprobs * np.ones(
            [input_start_ids.shape[0], 1]
        ).astype(np.bool_)
        beam_width = (num_beams * np.ones([input_start_ids.shape[0], 1])).astype(
            np.int32
        )
        start_ids = tokenizer.eos_token_id * np.ones(
            [input_start_ids.shape[0], 1]
        ).astype(np.int32)
        end_ids = tokenizer.eos_token_id * np.ones(
            [input_start_ids.shape[0], 1]
        ).astype(np.int32)

        # Not used
        stop_word_list = np.concatenate(
            [
                np.zeros([input_start_ids.shape[0], 1, 1]).astype(np.int32),
                (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32),
            ],
            axis=1,
        )

        bad_words_list = np.concatenate(
            [
                np.zeros([input_start_ids.shape[0], 1, 1]).astype(np.int32),
                (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32),
            ],
            axis=1,
        )

        streaming = [True]
        streaming_data = streaming * np.ones([input_start_ids.shape[0], 1]).astype(np.bool_)
        ret_gen_logits = [False]
        ret_gen_logits_data = ret_gen_logits * np.ones([input_start_ids.shape[0], 1]).astype(np.bool_)
        ret_context_logits = [False]
        ret_context_logits_data = ret_context_logits * np.ones([input_start_ids.shape[0], 1]).astype(np.bool_)
        inputs = {
            "input_ids": input_start_ids,
            "input_lengths": input_len,
            "request_output_len": output_len,
            "runtime_top_k": runtime_top_k,
            "runtime_top_p": runtime_top_p,
            #"beam_search_diversity_rate": beam_search_diversity_rate,
            "random_seed": random_seed,
            "temperature": temperature,
            "len_penalty": len_penalty,
            "repetition_penalty": repetition_penalty,
            "return_log_probs": is_return_log_probs,
            "beam_width": beam_width,
            #"start_id": start_ids,
            "end_id": end_ids,
            "streaming": streaming_data,
            # "return_generation_logits": ret_gen_logits_data,
            # "return_context_logits": ret_context_logits_data,
            # "bad_words_list": bad_words_list,
            # "stop_words_list": stop_word_list,
        }
        print('inputs', inputs)

        if (
            (use_trie or is_hard_strip)
            and allowed_sequences is not None
            and len(allowed_sequences) > 0
            and unstable_length > 0
        ):
            # It is not necessarily the same as unstable_length,
            # because we split the last token when it ends in space
            allowed_prefix_length = len(allowed_sequences[0])
            assert allowed_prefix_length <= MAX_ALLOWED_PREFIX_LENGTH

            for i in range(1, allowed_prefix_length + 1):
                allowed_prefixes = allowed_sequences[:, 0:i]
                # print('allowed_prefixes_step1', allowed_prefixes)
                # This actually hurts performance
                # allowed_prefixes = np.unique(allowed_prefixes, axis=0)
                offsets = np.arange(i, len(allowed_prefixes) * i + 1, i)
                # print('allowed_prefixes_step2', allowed_prefixes)
                allowed_prefixes = allowed_prefixes.flatten()
                # print('allowed_prefixes_step3', allowed_prefixes)
                offsets = np.concatenate(
                    (offsets, np.repeat(-1, (len(allowed_prefixes) - len(offsets))))
                )
                # print('allowed_prefixes_step4', offsets)
                allowed_prefixes = np.concatenate(([allowed_prefixes], [offsets]))
                # print('allowed_prefixes_step5', allowed_prefixes)
                allowed_prefixes = np.tile(
                    allowed_prefixes, (input_start_ids.shape[0], 1, 1)
                ).astype(np.int32)
                # print('allowed_prefixes_step6', allowed_prefixes)
                # inputs["allowed_prefixes_" + str(i)] = allowed_prefixes

        dim_to_squeeze = 0 if num_beams > 1 else 1

        stopping_condition_enabled, processor = get_processor(
            enable_stopping_condition,
            stop_nl,
            tokenizer,
            lang,
            prompt_len,
            unstable_text or "",
            orig_prompt,
            suffix,
        )
        print('before streaming_request')
        # if confs['use_new_model_scheme']:
        #     model_name = (
        #             "_".join(Path(self.config_path).parts[-2].split("_")[:-1])
        #             + "_model"
        #     )
        #
        (
            response,
            generated_tokens_length,
            reached_stopping_condition,
        ) = await streaming_request.execute(
            model_name=model_name,
            inputs=inputs,
            dim_to_squeeze=dim_to_squeeze,
            want_logprobs=want_logprobs,
            expand=expand,
            max_length=max_length,
            check_and_get_final_response=lambda res, finished: processor.process_and_get_final_response(
                res, finished
            ),
            get_max_tokens_reached_response=lambda res: processor.process_max_tokens_reached_response(res),
            token_handler=confs[
                "tokens_trie"
            ],
            language=lang,
            tokenizer=tokenizer,
            model_instance_name=self.model_instance_name,
            is_hard_strip=is_hard_strip,
            hard_strip_suf=hard_strip_suf,
        )
        print('response.generated', response.generated_length, flush=True)
        print('response.output', response.output, flush=True)
        generated_tokens = [
            out[prompt_len : prompt_len + generated]
            for generated, out in zip(response.generated_length, response.output)
        ]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        print('unstable_text', unstable_text, flush=True)
        print('generated_tokens', len(generated_tokens[0]), flush=True)
        print('decoded', decoded, flush=True)
        completions = []
        for i, (text, tokens, lps, g) in enumerate(
            zip(
                decoded,
                generated_tokens,
                response.lp_data,
                response.generated_length,
            )
        ):
            lps = lps.tolist()
            trim_size = 0
            if (use_trie or is_hard_strip) and unstable_text:
                # self.irrelevant_tokens = ~self.irrelevant_tokens
                # self.irrelevant_tokens[relevant_tokens] = False
                start_idx = text.find(unstable_text)
                print('start_idx', start_idx, flush=True)
                trim_size = len(unstable_text) if start_idx == 0 else 0
                print('trim_size', trim_size, flush=True)
                
                text = text[trim_size:]

            token_scores = [
                {
                    "text": tokenizer.decode([token], skip_special_tokens=True),
                    "score": score,
                }
                for idx, (token, score) in enumerate(zip(tokens, lps))
            ]

            tokens_to_remove = 0
            for token_score in token_scores:
                token_length = len(token_score["text"])
                if token_length <= trim_size:
                    tokens_to_remove += 1
                    trim_size -= token_length
                else:
                    token_score["text"] = token_score["text"][trim_size:]
                    break
            token_scores = token_scores[tokens_to_remove:]

            completions.append(
                {
                    "text": text,
                    # if not is_hard_strip
                    # else fix_completion_strip(text, indent_size, hard_strip_suf),
                    "score": sum(lps),
                    "token_scores": token_scores,
                }
            )

        if reached_stopping_condition or response.max_generated_length() == max_length:
            completions = processor.run_completions_postprocess(completions)

        response_json = {
            "prompt": {"text": "", "tokens": []},
            "stop_reason": "max_tokens",
            "requested_tokens": max_length,
            "generated_tokens": int(generated_tokens_length),
            "generated": completions,
            "infer_time_in_ms": 1000 * (time.time() - start_time),
            "generator_trims_prompt": False,
            "stopping_condition_enabled": stopping_condition_enabled,
            "trimmed_prompt_chars": trimmed_prompt,
            "time_diff_in_sec": time_diff,
            "prompt_tokens": len(input_start_ids[0]),
            "prefix_tokens": int(pre_tokens),
            "trimmed_prefix_tokens": int(trimmed_pre),
        }
        if hint_size > 0:
            response_json["hint_tokens"] = int(hint_size)
        elif is_hinting:
            response_json["trimmed_hint_tokens"] = len(hint_ids)

        if is_hinting:
            response_json["hint_chars"] = (
                len(sim_ref) + len(sim_next_lines) if sim_ref and sim_next_lines else 0
            )

        if use_fim:
            response_json["suffix_tokens"] = len(suffix_ids)
            response_json["trimmed_suffix_tokens"] = int(trimmed_suf)

        if imports_ids:
            response_json["imports_tokens"] = len(imports_ids)
            response_json["trimmed_imports_tokens"] = int(trimmed_imports)

        if exports_ids:
            response_json["exports_tokens"] = len(exports_ids)
            response_json["trimmed_exports_tokens"] = int(trimmed_exports)

        if query_payload.get("debug", False):
            output_prompt_tokens = [tokenizer.decode([i]) for i in input_start_ids[0]]
            response_json["prompt"] = {
                "text": "".join(output_prompt_tokens),
                "tokens": output_prompt_tokens,
            }
        tensor_output = pb_utils.Tensor(
            "output", np.array(json.dumps(response_json, indent=4), dtype=object)
        )
        responses.append(pb_utils.InferenceResponse(output_tensors=[tensor_output]))

    return responses
