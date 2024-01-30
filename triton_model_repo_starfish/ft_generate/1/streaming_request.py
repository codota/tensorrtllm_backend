#  Copyright (c) Tabnine LTD  2023.
#  All Rights Reserved.

import asyncio
import random
import time
from tritonclient import grpc, utils
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
        queues.unregister_all()
    else:
        # queues.put(result.get_response().id, result)
        print('recieved response')
        queues.put('1', result)


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
):
    # print(f"Starting inference for model {model_name}", flush=True)
    # print(f"Inputs: {inputs}", flush=True)
    # print(f"dim_to_squeeze: {dim_to_squeeze}", flush=True)
    # print(f"want_logprobs: {want_logprobs}", flush=True)
    # print(f"expand: {expand}", flush=True)
    # print(f"max_length: {max_length}", flush=True)
    # print(f"check_and_get_final_response: {check_and_get_final_response}", flush=True)
    # print(f"get_max_tokens_reached_response: {get_max_tokens_reached_response}", flush=True)
    # request_id = uuid.uuid4().hex
    queues.register('1')

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
        potential_unstable_tokens = inputs['input_ids'][0, -MAX_ALLOWED_PREFIX_LENGTH:]
        unstable_length, allowed_sequences = token_handler.get_allowed_sequences_for_prefix(
            potential_unstable_tokens,
            language.lower() not in ["python", "jupyter notebook"],
        )
        
        if unstable_length > 0:
            allowed_tokens = allowed_sequences[:,0]
            print("allowed_sequences", allowed_sequences)
            print("allowed_sequences[:,0]", allowed_sequences[:,0], flush=True)
            unstable_text = tokenizer.decode(input_start_ids[0, -unstable_length:])
            input_start_ids = input_start_ids[:, :-unstable_length]
            print("unstable_text", unstable_text, flush=True)
            len_of_unstable_text = len(unstable_text)
            print('len(unstable_text)', len(unstable_text), flush=True)
            print("input_start_ids", input_start_ids, flush=True)
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
                print('inputs_initial', inputs_initial, flush=True)
                input_list2 = [prepare_tensor(k, v) for k, v in inputs_initial.items()]
                output_list_initial = [grpc.InferRequestedOutput(name) for name in output_names_initial]
                get_client().async_stream_infer(model_name="tensorrt_llm_non_streaming", inputs=input_list2, outputs=output_list_initial, model_version="1")
                raw_response = await queues.get('1')
                output = raw_response.as_numpy("output_ids").squeeze(dim_to_squeeze)
                sequence_length = raw_response.as_numpy("sequence_length").squeeze(
                        dim_to_squeeze
                    )
                gen_logits = raw_response.as_numpy("generation_logits").squeeze(dim_to_squeeze)
                allowed = allowed_tokens
                allowed.sort()
                print("allowed", allowed, flush=True)
                print("gen_logits", gen_logits, flush=True)
                allowed_logits = gen_logits[:,0,allowed]
                print("allowed_logits", allowed_logits, flush=True)
                allowed_idx = np.argmax(allowed_logits)
                print("allowed_logits[allowed_udx]", allowed_logits[0][allowed_idx], flush=True)
                print("allowed_idx", allowed_idx, flush=True)
                allowed_val = allowed[allowed_idx]
                generated_unstable += tokenizer.decode(np.array([allowed_val]))
                print("generated_unstable", generated_unstable, flush=True)
                print("len_of_unstable_text", len_of_unstable_text, flush=True)
                if len(generated_unstable) < len_of_unstable_text:
                    inputs_initial['input_ids'] =  np.append(inputs_initial["input_ids"], np.array([allowed_val]), axis=1)
                    inputs_initial['input_lengths'] = np.add(inputs_initial["input_lengths"], 1)
                    
                    max_length = max_length - 1
                    curr_unstable_text = unstable_text[len_of_unstable_text - len(generated_unstable) +1:]
                    unstable_length, allowed_sequences = token_handler.get_allowed_sequences_for_prefix(
                        tokenizer.encode(curr_unstable_text),
                        language.lower() not in ["python", "jupyter notebook"],
                    )
                    allowed_tokens = allowed_sequences[:,0]
            inputs['input_ids'] =  np.append(inputs_initial["input_ids"], np.array([[allowed_val]]), axis=1)
            inputs['request_output_len'] = np.subtract(inputs["request_output_len"], np.subtract(inputs_initial['input_lengths'], inputs['input_lengths']))
            inputs['input_lengths'] = np.add(inputs_initial["input_lengths"], 1)
            
            output_len = inputs["request_output_len"]
            max_length = max_length - 1      
        # for k in inputs:
        #     if k.startswith("allowed_prefixes"):
        #         print(k, flush=True)
        #         print(inputs[k], flush=True)
        #         # allowed = inputs[k]
        #         allowed[k] = inputs[k]
        #         # del inputs[k]
        # if 'allowed_prefixes_1' in inputs:
        #     del inputs["allowed_prefixes_1"]
        # if 'allowed_prefixes_2' in inputs:
        #     del inputs["allowed_prefixes_2"]
        # res = token_handler.get_allowed_sequences_for_prefix(inputs['input_ids'][0,:])
        # print('res',res, flush=True)
        
        # split_req = True
        # if split_req:
        #     inputs_initial = copy.deepcopy(inputs)
        #     inputs_initial["request_output_len"] = np.ones_like(input_len).astype(np.int32)
        #     streaming = [False]
        #     streaming_data = streaming * np.ones([inputs['input_ids'].shape[0], 1]).astype(np.bool_)
        #     inputs_initial["streaming"] = streaming_data
        #     ret_gen_logits = [True]
        #     ret_gen_logits_data = ret_gen_logits * np.ones([inputs['input_ids'].shape[0], 1]).astype(np.bool_)
        #     inputs_initial["return_generation_logits"] = ret_gen_logits_data
        #     ret_context_logits = [True]
        #     ret_context_logits_data = ret_context_logits * np.ones([inputs['input_ids'].shape[0], 1]).astype(np.bool_)
        #     inputs_initial["return_context_logits"] = ret_context_logits_data
        #     print('inputs_initial', inputs_initial, flush=True)
        #     input_list2 = [prepare_tensor(k, v) for k, v in inputs_initial.items()]
        # del inputs["return_generation_logits"]
        # del inputs["return_context_logits"]
        # alt_input_list = [pb_utils.Tensor(k, v) for k, v in inputs_initial.items()]
        # object_methods = [method_name for method_name in dir(trtllm_request)
        #           if callable(getattr(trtllm_request, method_name))]
        # print(f'trtllm_request object_methods: {object_methods}', flush=True)
        # trtllm_responses = trtllm_request.exec(decoupled=True)
        # for trtllm_response in trtllm_responses:
        #     if trtllm_response.has_error():
        #                     raise pb_utils.TritonModelException(
        #                         trtllm_response.error().message())
        #     trtllm_output_tensors = trtllm_response.output_tensors()
        #     for i, trtllm_output_tensor in enumerate(trtllm_output_tensors):
        #         print(f'trtllm_output_tensor {i}: {trtllm_output_tensor.as_numpy()}', flush=True)
        # object_methods = [method_name for method_name in dir(trtllm_responses)
        #           if callable(getattr(trtllm_responses, method_name))]
        # print(f'trtllm_responses object_methods: {object_methods}', flush=True)
        # async_responses = trtllm_request.async_exec(decoupled=True)
        # object_methods = [method_name for method_name in dir(async_responses)
        #           if callable(getattr(async_responses, method_name))]
        # print(f'async_responses object_methods: {object_methods}', flush=True)
        # for i, async_resp in enumerate(await async_responses):
        #     print(f'async_resp {i}: {async_resp}', flush=True)
        #     async_responses.cancel()
        # for i, trtllm_response in enumerate(trtllm_responses):

        #             if trtllm_response.has_error():
        #                 raise pb_utils.TritonModelException(
        #                     trtllm_response.error().message())

        #             trtllm_output_tensors = trtllm_response.output_tensors()
        #             print(f'trtllm_output_tensors {i}: ', trtllm_output_tensors)
        #             for k in trtllm_output_tensors:
        #                 print(k.as_numpy(), flush=True)
        # print('trtllm_responses', trtllm_responses, flush=True)
            # output_list_initial = [grpc.InferRequestedOutput(name) for name in output_names_initial]
            # get_client().async_stream_infer(model_name="tensorrt_llm_non_streaming", inputs=input_list2, outputs=output_list_initial, model_version="1")
            # raw_response = await queues.get('1')
            # print('raw_response', raw_response, flush=True)
            # output = raw_response.as_numpy("output_ids").squeeze(dim_to_squeeze)
            # sequence_length = raw_response.as_numpy("sequence_length").squeeze(
            #         dim_to_squeeze
            #     )
            # gen_logits = raw_response.as_numpy("generation_logits").squeeze(dim_to_squeeze)
            # print('gen_logits', gen_logits[0][0], flush=True)
            # print('allowed', allowed, flush=True)
            # for k in allowed:
            #     print(k, allowed[k], flush=True)
            # allowed = allowed['allowed_prefixes_1'] + allowed['allowed_prefixes_2']
            # allowed.sort()
            # print('gen_logits', gen_logits, flush=True)
            # print('allowed_shape', allowed.shape, flush=True)
            # print('gen_logits_shape', gen_logits.shape, flush=True)
            # allowed_logits = gen_logits[:,0,allowed[:,0,:]]
            # print('allowed_logits', allowed_logits, flush=True)
            # allowed_idx = np.argmax(allowed_logits)
            # print('allowed_idx', allowed_idx, flush=True)
            # print('allowed_val', allowed[:,0, allowed_idx], flush=True)
            # allowed_val = allowed[:,0, allowed_idx]
            # print('sorted_allowed', sorted(allowed), flush=True)
            # print('max_val', np.argmax(gen_logits[0][0]), flush=True)
            # print('inputs_shape', inputs["input_ids"].shape, flush=True)
            # inputs['input_ids'] =  np.append(inputs["input_ids"], np.array([allowed_val]), axis=1)
            # inputs['input_lengths'] = np.add(inputs["input_lengths"], 1)
            # inputs['request_output_len'] = np.subtract(inputs["request_output_len"], 1)
            # output_len = inputs["request_output_len"]
            # max_length = max_length - 1
        ret_gen_logits = [False]
        ret_gen_logits_data = ret_gen_logits * np.ones([inputs['input_ids'].shape[0], 1]).astype(np.bool_)
        inputs["return_generation_logits"] = ret_gen_logits_data
        print("input_list", inputs, flush=True)
        print("output_names", output_names, flush=True)
        start = time.time()
        input_list = [prepare_tensor(k, v) for k, v in inputs.items()]
        output_list = [grpc.InferRequestedOutput(name) for name in output_names]
        get_client().async_stream_infer(
            model_name=model_name,
            inputs=input_list,
            outputs=output_list,
            # request_id=request_id,
            model_version="1",
        )
        print(f'get_client().async_stream_infer time: {time.time() - start}', flush=True)
        # print(f'inputs: {inputs}', flush=True)
        last_response = None
        generated_so_far = input_len[0]
        output_so_far = inputs["input_ids"]
        # for i, trtllm_response in enumerate(trtllm_responses):
        #             object_methods = [method_name for method_name in dir(trtllm_response) if callable(getattr(trtllm_response, method_name))]
        #             print(f'trtllm_single_response object_methods: {object_methods}', flush=True)
        #             if trtllm_response.has_error():
        #                 raise pb_utils.TritonModelException(
        #                     trtllm_response.error().message())

        #             trtllm_output_tensors = trtllm_response.output_tensors()
        #             # print(f'trtllm_output_tensors {i}: ', trtllm_output_tensors)
        #             output = trtllm_output_tensors[0].as_numpy().squeeze(dim_to_squeeze)
        #             output_so_far = np.append(output_so_far, output, axis=1)
        #             sequence_length = trtllm_output_tensors[1].as_numpy().squeeze(dim_to_squeeze)
        #             if want_logprobs:
        #                 lp_data = trtllm_output_tensors[2].as_numpy().squeeze(dim_to_squeeze)
        #             else:
        #                 lp_data = [None] * output_so_far.shape[0]
                    
        #             tokens = output[0]
        #             seq_len = sequence_length[0]
        #             in_len = input_len[0]
        #             out_len = output_len[0]
        #             end_id = end_ids[0]
        #             last_token = tokens[seq_len - 1]
        #             generated_so_far = generated_so_far + 1
        #             generated_length = generated_so_far - input_len.squeeze(dim_to_squeeze)
        #             all_done = not (generated_so_far - in_len < out_len and last_token not in end_id)
        #             last_response = execute_response.ExecuteResponse(
        #                 output_so_far, generated_length, lp_data
        #             )
        #             # all_done = seq_len - in_len >= out_len or last_token in end_id
        #             # print(f'end_id: {end_id}')
        #             # print(f'last_token: {last_token}')
        #             # print('all_done: ', all_done)
        #             # print('sequence_length: ', seq_len)
        #             # print('input_len: ', in_len)
        #             # print('output_len: ', out_len)
        #             # print('tokens: ', tokens)
        #             # print('generated_so_far: ', generated_so_far)
        #             final_response = check_and_get_final_response(last_response, False)
        #             # print(f'final_response: {final_response}')
        #             # print("last_response.max_generated_length()",last_response.max_generated_length())
        #             # print("max_length", max_length)
        #             if final_response:
        #                 #cancel_request(model_name, cancellation_token_id)
        #                 return final_response, last_response.max_generated_length(), True

        #             if all_done:
        #                 #cancel_request(model_name, cancellation_token_id)
        #                 if max_length == last_response.max_generated_length():
        #                     # print(f"max_length 1: {last_response.max_generated_length()}")
        #                     final_response = get_max_tokens_reached_response(last_response)
        #                 else:
        #                     # print(f"max_length 2: {last_response.max_generated_length()}")
        #                     final_response = check_and_get_final_response(last_response, True)
        #                 # print(f'final final_response: {final_response}')
        #                 if final_response:
        #                     return final_response, last_response.max_generated_length(), True
        #                 break

        while True:
            start = time.time()
            raw_response = await queues.get('1')
            print(f'queues.get(1) time: {time.time() - start}', flush=True)
            if raw_response is None:
                break
            output = raw_response.as_numpy("output_ids").squeeze(dim_to_squeeze)
            # add output to the output_so_far numpy array
            print('output_so_far: ', output_so_far)
            print('output: ', output)
            output_so_far = np.append(output_so_far, output, axis=1)
            print('output_so_far_after: ', output_so_far)

            sequence_length = raw_response.as_numpy("sequence_length").squeeze(
                dim_to_squeeze
            )

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
            # all_done = seq_len - in_len >= out_len or last_token in end_id
            print(f'end_id: {end_id}')
            print(f'last_token: {last_token}')
            print('all_done: ', all_done)
            print('sequence_length: ', seq_len)
            print('input_len: ', in_len)
            print('output_len: ', out_len)
            print('tokens: ', tokens)
            print('generated_so_far: ', generated_so_far)
            final_response = check_and_get_final_response(last_response, False)
            print(f'final_response: {final_response}')
            print("last_response.max_generated_length()",last_response.max_generated_length())
            print("max_length", max_length)
            if final_response:
                #cancel_request(model_name, cancellation_token_id)
                get_client()._stream._response_iterator.cancel()
                return final_response, last_response.max_generated_length(), True

            if all_done:
                #cancel_request(model_name, cancellation_token_id)
                get_client()._stream._response_iterator.cancel()
                if max_length == last_response.max_generated_length():
                    print(f"max_length 1: {last_response.max_generated_length()}")
                    final_response = get_max_tokens_reached_response(last_response)
                else:
                    print(f"max_length 2: {last_response.max_generated_length()}")
                    final_response = check_and_get_final_response(last_response, True)
                print(f'final final_response: {final_response}')
                if final_response:
                    return final_response, last_response.max_generated_length(), True
                break
        print('output_so_far: ', output_so_far)
        print(tokenizer.decode(output_so_far[0], skip_special_tokens=True))
        return last_response, last_response.max_generated_length(), False
    except Exception as e:
        print('Exception')
        print(traceback.format_exc())
    finally:
        print('Cleaning up...')
        queues.unregister('1')


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
