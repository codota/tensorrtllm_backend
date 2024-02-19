#!/usr/bin/python

from fastapi import FastAPI
from typing import Optional, List
from pydantic import BaseModel
import tritonclient.http.aio as httpclient
import tritonclient.grpc as grpcclient
import numpy as np
from tritonclient.utils import np_to_triton_dtype
import os
import sys
from functools import partial
import queue

import numpy as np
from tritonclient.utils import InferenceServerException
import time
from sse_starlette.sse import EventSourceResponse
import uuid
from queues import Queues
import asyncio
import random
import tritonclient.http

triton_client = httpclient.InferenceServerClient(
    url="127.0.0.1:8000", verbose=False
)


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

app = FastAPI()

@app.get("/health")
async def health():
    return {"ok": True}

url_llama = 'localhost:8001' #llama-trtllm.triton:8001
url_star = 'localhost:8001' #chat-trtllm.triton:8001
client = None
queues = Queues()

client_llama = None
client_starchat = None

def stream_callback(result, error):
    if error is not None:
        print(f"Got error from grpc: {error}")
        queues.unregister_all()
    else:
        queues.put(result.get_response().id, result)
        if result.get_response().parameters['triton_final_response'].bool_param:
            queues.put(result.get_response().id, None)

def get_client():
    global client
    url = '34.135.21.21:8001'
    if client is None:
        try:
            client = grpcclient.InferenceServerClient(url=url)
            client.start_stream(stream_callback)

        except Exception as e:
            print("client creation failed: " + str(e))
            sys.exit(1)
    return client

def close_client():
    global client
    if client is not None:
        queues.unregister_all()
        client.stop_stream()
        client.close()
        client = None

protocol = 'grpc'

class Parameters(BaseModel):
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 1.0
    seed: Optional[int] = 42
    num_beams: Optional[int] = 1
    top_k: Optional[int] = 1
    
class Request(BaseModel):
    inputs: str
    parameters: Optional[Parameters]
    

def prepare_tensor(name, input, protocol="http"):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(name, input.shape,
                               np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t   

def prep_inputs(request: Request):
    global client
    request_id = str(uuid.uuid1().int>>64) # uuid.uuid4().hex
    #print(request_id)
    queues.register(request_id)
    model_name = "ensemble"
    input0 = [[request.inputs]]
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.int32) * request.parameters.max_new_tokens
    streaming = [[True]] # False if not streaming
    streaming_data = np.array(streaming, dtype=bool)
    temperature = np.ones_like(input0).astype(np.float32) * request.parameters.temperature
    beams = np.ones_like(input0).astype(np.int32) * request.parameters.num_beams
    top_k = np.ones_like(input0).astype(np.int32) * request.parameters.top_k
    end_id = np.ones_like(input0).astype(np.int32) * 2
    random_seed_data = np.ones_like(input0).astype(np.uint64) * request.parameters.seed

    inputs = [
        prepare_tensor("text_input", input0_data,protocol),
        prepare_tensor("max_tokens", output0_len, protocol),
        prepare_tensor("stream", streaming_data, protocol),
        prepare_tensor("beam_width", beams, protocol),
        prepare_tensor("end_id", end_id, protocol),
        prepare_tensor("random_seed", random_seed_data, protocol),
        prepare_tensor("top_k", top_k, protocol),
        prepare_tensor("temperature", temperature, protocol),
    ]
    # if request.parameters.seed:
    #     random_seed = np.ones_like(input0).astype(np.uint64) * request.parameters.seed
    #     inputs.append(prepare_tensor("random_seed", random_seed, protocol))

    # Send request
    # start_time = time.time()
    # print('before async_stream_infer')
    client = get_client()
    client.async_stream_infer(model_name, inputs, request_id=request_id, enable_empty_final_response=True)
    # print('after async_stream_infer')
    # print(time.time() - start_time)
    return request_id

# @app.post("/generate")
# async def generate(request: Request):
    
#     try:
#         request_id = prep_inputs(request)
#         all_the_data = ''
#         while True:
#             raw_response = await queues.get(request_id)
#             if raw_response is None:
#                 break
#             m = raw_response.as_numpy("text_output")
#             print(request_id+'!', m)
#             all_the_data += m[0].decode('utf8')
#         return  {'generated_text': all_the_data}
    
#     except Exception as e:
#         print(e)
#         return {'generated_text': all_the_data}
    
@app.post("/generate")
async def generate(request: Request):
    inputs = {}
    try:
        input0 = [[request.inputs]]
        inputs["text_input"] = np.array(input0).astype(object)
        inputs["max_tokens"] = np.ones_like(input0).astype(np.int32) * request.parameters.max_new_tokens
        streaming = [[False]] # False if not streaming
        inputs["stream"] = np.array(streaming, dtype=bool)
        inputs["beam_width"] = np.ones_like(input0).astype(np.int32) * request.parameters.num_beams
        inputs["end_id"] = np.ones_like(input0).astype(np.int32) * 0
        inputs["random_seed"] = np.ones_like(input0).astype(np.uint64) * request.parameters.seed
        inputs["top_k"] = np.ones_like(input0).astype(np.int32) * request.parameters.top_k
        inputs["temperature"] = np.ones_like(input0).astype(np.float32) * request.parameters.temperature
        input_list = [prepare_tensor(k, v,) for k, v in inputs.items()]
        output_list = [httpclient.InferRequestedOutput(name) for name in ['text_output']]
        start_time = time.time()
        result = await triton_client.infer(model_name="ensemble", inputs=input_list, outputs=output_list)
        print(time.time() - start_time)
        return {'generated_text': result.as_numpy("text_output")[0].decode('utf8')}
        
    except Exception as e:
        print(e)

@app.post("/generate_stream")
async def generate_stream(request: Request):
    request_id = prep_inputs(request)
    start = time.time()

    async def gen_data():
        while True:
            try:
                result = await queues.get(request_id)
                # result = await queues.get(request_id)
                if result is None or result.as_numpy("text_output") is None:
                    print(time.time() - start)
                    break
            except Exception:
                print(time.time() - start)
                break
            if type(result) == InferenceServerException:
                print("Received an error from server:")
                print(result)
            else:
                m = result.as_numpy("text_output")
                print(m[0].decode('utf8'))
                yield m[0].decode('utf8')    
    return EventSourceResponse(gen_data())


#TODO: run with  uvicorn qyery_trtllm:app --port 8081