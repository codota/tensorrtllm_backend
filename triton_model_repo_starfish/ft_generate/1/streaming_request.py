#  Copyright (c) Tabnine LTD  2023.
#  All Rights Reserved.

import asyncio
import random
import time
from tritonclient import grpc, utils
import tritonclient.grpc as grpcclient
from queues import Queues
from tok_trie import TokenHandler
import uuid
import execute_response
import numpy as np
import threading
import traceback
import triton_python_backend_utils as pb_utils
import copy

queues = Queues()

client = None
MAX_ALLOWED_PREFIX_LENGTH = 5


def get_client():
    global client
    if client is None:
        client = grpc.InferenceServerClient("127.0.0.1:8001")
        client.start_stream(stream_callback)

    return client


def close_client():
    global client
    if client is not None:
        queues.unregister_all()
        client.stop_stream()
        client.close()
        client = None


def stream_callback(result, error):
    if error is not None:
        print(f"Got error from grpc: {error}")
        print(result)
        print(dir(result))
        queues.unregister_all()
    else:
        # queues.put(result.get_response().id, result)
        print('recieved response')
        print(result.get_response())
        print(dir(result.get_response()))
        print(result.get_response().model_name)
        queues.put(result.get_response().id, result)


async def execute(
    model_name,
    inputs,
    dim_to_squeeze,
    want_logprobs,
    expand,
    max_length,
    check_and_get_final_response,
    get_max_tokens_reached_response,
    token_handler: TokenHandler,
    language: str,
    tokenizer,
    model_instance_name: str,
    is_hard_strip: bool,
    hard_strip_suf: str,
):
    request_id = str(int(model_instance_name.split('_')[-1]) + 1)
    queues.register(request_id)
    queues.register(request_id+'1')
    print('request_id', request_id)
    try:
        input_len = inputs["input_lengths"]
        output_len = inputs["request_output_len"]
        end_ids = inputs["end_id"]
        output_names_initial = ["output_ids", "sequence_length", "generation_logits", "context_logits"]
        output_names = ["output_ids", "sequence_length"]
        if want_logprobs:
            output_names.append("output_log_probs")
        allowed = {}
        generated_unstable = ''
        input_start_ids = inputs['input_ids']
        if is_hard_strip:
            print('hard_strip_suf', hard_strip_suf, flush=True)
            potential_unstable_tokens = tokenizer.encode(hard_strip_suf)
            print('hard_strip potential_unstable_tokens', potential_unstable_tokens)
            unstable_length, allowed_sequences = token_handler.get_allowed_sequences_for_prefix(potential_unstable_tokens, False, True)
            allowed_sequences = allowed_sequences.astype(np.int32)
            # unstable_text = hard_strip_suf
        else:
            potential_unstable_tokens = inputs['input_ids'][0, -MAX_ALLOWED_PREFIX_LENGTH:]
            unstable_length, allowed_sequences = token_handler.get_allowed_sequences_for_prefix(
                potential_unstable_tokens,
                language.lower() not in ["python", "jupyter notebook"],
            )
        # print('potential_unstable_tokens', potential_unstable_tokens, flush=True)
        print('unstable_length', unstable_length, flush=True)

        if unstable_length > 0:
            allowed_tokens = allowed_sequences[:,0]
            # print('allowed_tokens', allowed_tokens, flush=True)
            if is_hard_strip:
                unstable_text = hard_strip_suf
                print('hard_strip_vals')
                for c in hard_strip_suf:
                    print(ord(c))
            else:
                unstable_text = tokenizer.decode(input_start_ids[0, -unstable_length:])
                input_start_ids = input_start_ids[:, :-unstable_length]

            print('unstable_text', unstable_text, flush=True)
            # print('stable_text', tokenizer.decode(input_start_ids[0, :-unstable_length]), flush=True)
            
            # input_start_ids = input_start_ids[:, :-unstable_length]
            len_of_unstable_text = len(unstable_text)
            if len(input_start_ids[0]) == 0:
                raise ValueError("empty prompt was given")
            inputs_initial = copy.deepcopy(inputs)
            inputs_initial["input_ids"] = input_start_ids
            inputs_initial["input_lengths"] = np.subtract(inputs["input_lengths"], unstable_length)
            inputs_initial["request_output_len"] = np.ones_like(input_len).astype(np.int32)
            streaming = [False]
            streaming_data = streaming * np.ones([inputs['input_ids'].shape[0], 1]).astype(np.bool_)
            inputs_initial["streaming"] = streaming_data
            ret_gen_logits = [True]
            ret_gen_logits_data = ret_gen_logits * np.ones([inputs['input_ids'].shape[0], 1]).astype(np.bool_)
            inputs_initial["return_generation_logits"] = ret_gen_logits_data
            while len(generated_unstable) < len_of_unstable_text:
                print('len(generated_unstable)', len(generated_unstable), flush=True)
                print('len_of_unstable_text', len_of_unstable_text, flush=True)
                print("inputs_initial", tokenizer.decode(inputs_initial['input_ids'][0]), flush=True)
                print("inputs_initial_token_ids", inputs_initial["input_ids"][0], flush=True)
                input_list2 = [prepare_tensor(k, v) for k, v in inputs_initial.items()]
                output_list_initial = [grpc.InferRequestedOutput(name) for name in output_names_initial]
                get_client().async_stream_infer(model_name="tensorrt_llm_non_streaming", inputs=input_list2, outputs=output_list_initial, model_version="1", request_id=request_id+'1')
                # raw_response = get_client().infer(model_name="tensorrt_llm_non_streaming", inputs=input_list2, outputs=output_list_initial, model_version="1")
                raw_response = await queues.get(request_id+'1')
                output = raw_response.as_numpy("output_ids").squeeze(dim_to_squeeze)
                sequence_length = raw_response.as_numpy("sequence_length").squeeze(dim_to_squeeze)
                print('output', output, flush=True)
                print('sequence_length', sequence_length, flush=True)
                print('gen_logits', raw_response.as_numpy("generation_logits"), flush=True)

                gen_logits = raw_response.as_numpy("generation_logits").squeeze(dim_to_squeeze)
                allowed = allowed_tokens
                allowed.sort()
                print('allowed', allowed, flush=True)
                allowed_logits = gen_logits[:,0,allowed]
                allowed_idx = np.argmax(allowed_logits)
                allowed_val = allowed[allowed_idx]
                print('allowed_val', allowed_val, flush=True)
                # print('allowed_val.shape', allowed_val.shape, flush=True)
                # print('inputs')
                generated_unstable += tokenizer.decode(np.array([allowed_val]))
                if len(generated_unstable) < len_of_unstable_text:
                    inputs_initial['input_ids'] =  np.append(inputs_initial["input_ids"], np.array([[allowed_val]]), axis=1)
                    print('inputs_initial["input_ids"]', inputs_initial["input_ids"])
                    inputs_initial['input_lengths'] = np.add(inputs_initial["input_lengths"], 1)
                    
                    max_length = max_length - 1
                    curr_unstable_text = unstable_text[len_of_unstable_text - len(generated_unstable) +1:]
                    if is_hard_strip:
                        print('allowed_sequences', allowed_sequences, flush=True)
                        allowed_sequences = allowed_sequences[allowed_sequences[:,0] == allowed_val][:,1:]
                        # unstable_length, allowed_sequences = token_handler.get_allowed_sequences_for_prefix(
                        #     tokenizer.encode(curr_unstable_text),
                        #     False,
                        #     True
                        # )
                        allowed_sequences = allowed_sequences.astype(np.int32)
                    else:
                        unstable_length, allowed_sequences = token_handler.get_allowed_sequences_for_prefix(
                            tokenizer.encode(curr_unstable_text),
                            language.lower() not in ["python", "jupyter notebook"],
                        )
                    allowed_tokens = allowed_sequences[:,0]
            inputs['input_ids'] =  np.append(inputs_initial["input_ids"], np.array([[allowed_val]]), axis=1)
            print('inputs["input_ids"] after while', inputs["input_ids"])
            print('inputs["input_ids"] after while decoded', tokenizer.decode(inputs['input_ids'][0]), flush=True)
            inputs['request_output_len'] = np.subtract(inputs["request_output_len"], np.subtract(inputs_initial['input_lengths'], inputs['input_lengths']))
            inputs['input_lengths'] = np.add(inputs_initial["input_lengths"], 1)
            
            output_len = inputs["request_output_len"]
            max_length = max_length - 1      
        ret_gen_logits = [False]
        ret_gen_logits_data = ret_gen_logits * np.ones([inputs['input_ids'].shape[0], 1]).astype(np.bool_)
        inputs["return_generation_logits"] = ret_gen_logits_data
        start = time.time()
        input_list = [prepare_tensor(k, v) for k, v in inputs.items()]
        output_list = [grpc.InferRequestedOutput(name) for name in output_names]
        get_client().async_stream_infer(
            model_name=model_name,
            inputs=input_list,
            outputs=output_list,
            model_version="1",
            request_id=request_id
        )
        last_response = None
        generated_so_far = input_len[0]
        output_so_far = inputs["input_ids"]
        print('tokenizer.decode(inputs["input_ids"][0])', tokenizer.decode(inputs['input_ids'][0]),)
        while True:
            start = time.time()
            raw_response = await queues.get(request_id)
            if raw_response is None:
                break
            output = raw_response.as_numpy("output_ids").squeeze(dim_to_squeeze)
            # add output to the output_so_far numpy array
            output_so_far = np.append(output_so_far, output, axis=1)
            sequence_length = raw_response.as_numpy("sequence_length").squeeze(dim_to_squeeze)
            if want_logprobs:
                lp_data = raw_response.as_numpy("output_log_probs").squeeze(
                    dim_to_squeeze
                )
            else:
                lp_data = [None] * output_so_far.shape[0]
            # check all done on first beam only
            tokens = output[0]
            seq_len = sequence_length[0]
            in_len = input_len[0]
            out_len = output_len[0]
            end_id = end_ids[0]
            last_token = tokens[seq_len - 1]
            generated_so_far = generated_so_far + 1
            generated_length = generated_so_far - input_len.squeeze(dim_to_squeeze)
            all_done = not (generated_so_far - in_len < out_len and last_token not in end_id)
            last_response = execute_response.ExecuteResponse(
                output_so_far, generated_length, lp_data
            )
            print('last_response.output', last_response.output, flush=True)
            print('tokenizer.decode(last_response.output[0])', tokenizer.decode(last_response.output[0]) , flush=True)
            final_response = check_and_get_final_response(last_response, False)
            if generated_length > 50:
                print('canceling request')
                # queues.unregister('1')
                # get_client()._stream._response_iterator.cancel()
                # print('shmuf',dir(get_client()._stream._response_iterator))
                cancel_inputs = [
                    grpcclient.InferInput('input_ids', [1, 1], "INT32"),
                    grpcclient.InferInput('input_lengths', [1, 1], "INT32"),
                    grpcclient.InferInput('request_output_len', [1, 1], "INT32"),
                    grpcclient.InferInput('stop', [1, 1], "BOOL"),
                ]

                cancel_inputs[0].set_data_from_numpy(np.empty([1, 1], dtype=np.int32))
                cancel_inputs[1].set_data_from_numpy(np.zeros([1, 1], dtype=np.int32))
                cancel_inputs[2].set_data_from_numpy(np.array([[0]], dtype=np.int32))
                cancel_inputs[3].set_data_from_numpy(np.array([[True]], dtype='bool'))
                print('request_id1',request_id)
                get_client().async_stream_infer(
                            'tensorrt_llm',
                            cancel_inputs,
                            request_id=request_id,
                            parameters={'Streaming': True})
                # time.sleep(2)
                break

            if final_response:
                # print('canceling request')
                # # queues.unregister('1')
                # # get_client()._stream._response_iterator.cancel()
                # # print('shmuf',dir(get_client()._stream._response_iterator))
                # cancel_inputs = [
                #     grpcclient.InferInput('input_ids', [1, 1], "INT32"),
                #     grpcclient.InferInput('input_lengths', [1, 1], "INT32"),
                #     grpcclient.InferInput('request_output_len', [1, 1], "INT32"),
                #     grpcclient.InferInput('stop', [1, 1], "BOOL"),
                # ]

                # cancel_inputs[0].set_data_from_numpy(np.empty([1, 1], dtype=np.int32))
                # cancel_inputs[1].set_data_from_numpy(np.zeros([1, 1], dtype=np.int32))
                # cancel_inputs[2].set_data_from_numpy(np.array([[0]], dtype=np.int32))
                # cancel_inputs[3].set_data_from_numpy(np.array([[True]], dtype='bool'))
                # print('request_id2',request_id)
                # get_client().async_stream_infer(
                #             'tensorrt_llm',
                #             cancel_inputs,
                #             request_id=request_id,
                #             parameters={'Streaming': True})
                # time.sleep(2)
                return final_response, last_response.max_generated_length(), True

            if all_done:
                # print('canceling request')
                # # queues.unregister('1')
                # cancel_inputs = [
                #     grpcclient.InferInput('input_ids', [1, 1], "INT32"),
                #     grpcclient.InferInput('input_lengths', [1, 1], "INT32"),
                #     grpcclient.InferInput('request_output_len', [1, 1], "INT32"),
                #     grpcclient.InferInput('stop', [1, 1], "BOOL"),
                # ]

                # cancel_inputs[0].set_data_from_numpy(np.empty([1, 1], dtype=np.int32))
                # cancel_inputs[1].set_data_from_numpy(np.zeros([1, 1], dtype=np.int32))
                # cancel_inputs[2].set_data_from_numpy(np.array([[0]], dtype=np.int32))
                # cancel_inputs[3].set_data_from_numpy(np.array([[True]], dtype='bool'))
                # print('request_id3',request_id)
                # get_client().async_stream_infer(
                #             'tensorrt_llm',
                #             cancel_inputs,
                #             request_id=request_id,
                #             parameters={'Streaming': True})
                # time.sleep(2)
                if max_length == last_response.max_generated_length():
                    final_response = get_max_tokens_reached_response(last_response)
                else:
                    final_response = check_and_get_final_response(last_response, True)
                if final_response:
                    return final_response, last_response.max_generated_length(), True
                break
        print(tokenizer.decode(output_so_far[0], skip_special_tokens=True))
        return last_response, last_response.max_generated_length(), False
    except Exception as e:
        print('Exception')
        print(traceback.format_exc())
    finally:
        print('Cleaning up...')
        queues.unregister(request_id)
        queues.unregister(request_id+'1')


def prepare_tensor(name, input):
    t = grpc.InferInput(name, input.shape, utils.np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def cancel_request(model_name, cancellation_token_id):
    inputs = {
        #"action": np.array([[1]], dtype=np.uint32),
        #"cancellation_token_id": np.array([[cancellation_token_id]], dtype=np.uint64),
        "input_ids": np.array([[]], dtype=np.int32),
        "input_lengths": np.array([[0]], dtype=np.int32),
        "request_output_len": np.array([[]], dtype=np.int32),
    }
    input_list = [prepare_tensor(k, v) for k, v in inputs.items()]

    get_client().async_stream_infer(
        model_name=model_name,
        inputs=input_list,
        outputs=[],
        model_version="1",
    )


current_cancellation_token_lock = threading.Lock()
current_cancellation_token_id = 0


def next_cancellation_token_id():
    current_cancellation_token_lock.acquire()

    try:
        global current_cancellation_token_id
        current_cancellation_token_id = current_cancellation_token_id + 1
        return current_cancellation_token_id
    finally:
        current_cancellation_token_lock.release()
