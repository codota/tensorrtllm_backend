import random
import time
from locust import HttpUser, task, between
import datasets
from sseclient import SSEClient


problems = datasets.load_dataset(
        "nuprl/MultiPL-E", f"humaneval-py"
    )

class QuickstartUser(HttpUser):

    # @task
    # def hello_world(self):
    #     prompt = 'ii'*4000 # problems["test"][10]["prompt"] #random.choice(problems["test"])["prompt"]
    #     # prompt = random.choice(problems["test"])["prompt"]
    #     start_time = time.time()
    #     if prompt.endswith('\n'):
    #         prompt = prompt[:-1]
    #     data = {
    #         "inputs": prompt,
    #         "parameters": {
    #             "do_sample": True,
    #             "max_new_tokens": 100,
    #             "seed": 42,
    #             "num_beams": 1,
    #             "top_k": 1
    #         },
    #     }
    #     print(prompt)
    #     r = self.client.post("/generate", json=data)
    #     print(time.time() - start_time)
    #     print(r.json()["generated_text"])
    #     return r.json()["generated_text"]

    @task
    def hello_world_stream(self):
        prompt = 'ii'*4000 # problems["test"][10]["prompt"] #random.choice(problems["test"])["prompt"]
        prompt = random.choice(problems["test"])["prompt"]
        # prompt = f"def prin"
        if prompt.endswith('\n'):
            prompt = prompt[:-1]
        data = {
            "inputs": prompt,
            "parameters": {
                "do_sample": False,
                "max_new_tokens": 100,
                "num_beams": 1,

            },
            # "model_name": "tensorrt_llm"
        }
        print(prompt)
        # r = self.client.post("/generate_stream", json=data)
        with self.client.post("/generate_stream",  json=data, stream=False, catch_response=True) as response:
            # client = SSEClient(response.iter_lines())
            # for event in client.events():
            #     print(f"Event: {event.event}")
            #     print(f"Data: {event.data}")
            if response.status_code == 200:
                if "failed" in response.text:
                    response.failure(response.text)
                else:
                    response.success()
                    # print(response.text)
            else:
                response.failure(response.text)
            
        # self.handle_response(r)
            # client = SSEClient(response.iter_lines())
            # for event in client.events():
            #     print(f"Event: {event.event}")
            #     print(f"Data: {event.data}")
            #     if "failed" in event.data:
            #         raise Exception(event.data)
            #     print("---")
        # return r.content