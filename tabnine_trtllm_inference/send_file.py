import requests
import json
import sseclient


url = 'http://localhost:8083/generate_stream'

headers = {
    'Accept': 'text/event-stream',
    'Content-Type': 'application/json'
}
text = 'def truncate_number(number: float) -> float:\n    """ Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    """\n'
# text = 'def print_hello_world():'
data = {
    "inputs": text,
    "parameters": {
        "num_beams": 1,
        "language":"python",
        "stop_nl":False,
        "enable_stopping_condition": False
    },
    "model_name": "fastertransformer",
}

data = {
    "inputs": text,
  "parameters": {
    "max_new_tokens": 200,
    "temperature": 1,
    "top_p": 0.8,
    "seed": 42,
    "details": True,
    "stop": [
      "</s>",
      "<|END|>"
    ],
    "num_beams": 1,
    "top_k": 0,
    "len_penalty": 1,
    "repetition_penalty": 1
  },
}

request = requests.post(url, headers=headers, json=data, stream=True)
client = sseclient.SSEClient(request)
for event in client.events():
        if event.data != '[DONE]':
            print(event.data, end="", flush=True)

# print(response.json())