
import asyncio


class Queues:
    def __init__(self):
        self.queues = {}

    def register(self, queue_id):
        self.queues[queue_id] = {
            "queue": asyncio.Queue(),
            "loop": asyncio.get_running_loop(),
        }

    def unregister(self, queue_id):
        self.put(queue_id, None)
        if queue_id in self.queues:
            self.queues.pop(queue_id)

    def unregister_all(self):
        for queue_id in list(self.queues.keys()):
            self.unregister(queue_id)

    def put(self, queue_id, item):
        queue = self.queues.get(queue_id)
        if not queue:
            return

        # asyncio.run_coroutine_threadsafe(queue["queue"].put(item), queue["loop"])
        # asyncio.run(queue["queue"].put(item))
        asyncio.run_coroutine_threadsafe(queue["queue"].put(item), queue["loop"])

    async def get(self, queue_id):
        queue = self.queues.get(queue_id)
        if not queue:
            return None

        return await queue["queue"].get()