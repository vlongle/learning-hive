import ray
from time import sleep

@ray.remote
class Counter:
    def __init__(self):
        for _ in range(5):
            sleep(1)
        self.value = 0
        print('Counter created!')

    def increment(self):
        self.value += 1
        return self.value

    def get_counter(self):
        return self.value

# Create an actor from this class.
counter = Counter.remote()
print(ray.get(counter.get_counter.remote()))
print('END')