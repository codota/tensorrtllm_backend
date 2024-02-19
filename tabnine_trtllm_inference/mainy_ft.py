import time
import traceback
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import tritonclient.http
import tritonclient.http.aio as httpclient
import numpy as np
from tritonclient.utils import np_to_triton_dtype
import srsly
import aiohttp
import json
import logging

app = FastAPI()

triton_client = httpclient.InferenceServerClient(url="127.0.0.1:8000", verbose=True)
generator_name = "ft_generate"
generator_version = "1"


class Parameters(BaseModel):
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.2
    seed: Optional[int] = None
    details: Optional[bool] = True
    stop: Optional[List] = [""] # "</s>", "<|end|>" ]
    num_beams: Optional[int] = 1
    enable_stopping_condition: Optional[bool] = False
    stop_nl: Optional[bool] = False

class Request(BaseModel):
    inputs: str = "def print_hello_world():"
    parameters: Optional[Parameters]
    model_name: Optional[str] = 'fastertransformer'

def prepare_tensor(name, input, protocol):
    client_util = httpclient
    t = client_util.InferInput(name, input.shape,
                               np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

model_metadata = triton_client.get_model_metadata(
            model_name=generator_name, model_version=generator_version
            )
model_config = triton_client.get_model_config(
            model_name=generator_name, model_version=generator_version
            )

logging.info("model metadata: %s", model_metadata)
logging.info("model config: %s", model_config)

@app.post("/generate_stream")
async def generate(request: Request):
    try:
        start = time.time()
        
        prompt = request.inputs
        if prompt.endswith('\n'):
                prompt = prompt[:-1]
        # prompt = (prompt + " ")*10
        query = httpclient.InferInput(name="TEXT", shape=(1,), datatype="BYTES")
        model_score = httpclient.InferRequestedOutput(name="output", binary_data=False)
        request_payload = json.dumps(
            {
            "prompt": prompt,
            "num_beams": request.parameters.num_beams,
            "stop_nl": False,
            'indent': None,
            'debug': False,
            # "lang_token": False,
            "language": "python",
            "enable_stopping_condition": request.parameters.enable_stopping_condition,
            'preceding_text': None, 
            'source': None,
    #         "enrichment": ench,
            "model_name": request.model_name,
            "stop_nl": request.parameters.stop_nl,
            "max_length": request.parameters.max_new_tokens,
            # "num_return_sequences": request.parameters.num_beams
        }

        )
        query.set_data_from_numpy(np.asarray([request_payload], dtype=object))
        #input0 = [[prompt]]
        #input0_data = np.array(input0).astype(object)
        #output0_len = np.ones_like(input0).astype(np.uint32) * 300

        #inputs = [
        #    prepare_tensor("INPUT_0", input0_data, "http"),
        #    prepare_tensor("INPUT_1", output0_len, "http"),
        #]
        result = await triton_client.infer(model_name=generator_name, model_version=generator_version, inputs=[query], outputs=[model_score], timeout=15)

        # data = {
        #         "inputs": prompt,
        #         "parameters": {
        #             "do_sample": False,
        #             "max_new_tokens": 300,
        #             "num_beams": 1
        #         },
        #     }
        # async with aiohttp.ClientSession('http://localhost:8080') as session:

        #     async with session.post("/generate", json=data) as resp:
        #         x = await resp.json()
        #         return x
            

        # generated_text_list = [result.as_numpy('OUTPUT_0')[0].decode()[len(prompt):], result.as_numpy('OUTPUT_0')[0].decode()[len(prompt):]]

        # x = result.as_numpy('OUTPUT_0')[0].decode()[len(prompt):]
        # srsly.write_jsonl('prompt_lines_trt.jsonl',[{'prompt': prompt, 'output': x, 'time': time.time() - start}], append=True)
        # print(len(x.split())
        json_res = json.loads(result.as_numpy("output").tolist()[:])
        generated_text_list = [e['text'] for e in json_res['generated']]
        print(result.as_numpy("output"))
    except Exception as e:
        print('Exception')
        print(traceback.format_exc())
    return {"generated_text": generated_text_list[0]}
