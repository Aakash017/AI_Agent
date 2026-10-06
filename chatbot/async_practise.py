import asyncio
import time


async def lookup(name):
    print(f"Starting {name}")
    # await asyncio.sleep(2)  # Simulate waiting for a service
    time.sleep(2)
    print(f"Finished {name}")
    return name


async def main():
    start = time.perf_counter()

    await lookup("Energy")
    await lookup("Order")

    print(f"Sequential: {time.perf_counter() - start:.1f}s")

    start = time.perf_counter()

    await asyncio.gather(
        lookup("Energy"),
        lookup("Order"),
    )

    print(f"Concurrent: {time.perf_counter() - start:.1f}s")


asyncio.run(main())