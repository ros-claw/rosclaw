"""Entry point: python -m rosclaw.dashboard"""

import asyncio

from .metrics import DashboardMetrics
from .server import DashboardServer


async def main():
    metrics = DashboardMetrics()
    server = DashboardServer(metrics, host="0.0.0.0", port=8765)
    await server.start()
    print(f"DashboardServer started on http://0.0.0.0:8765")
    print("Press Ctrl+C to stop")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await server.stop()
        print("DashboardServer stopped")


if __name__ == "__main__":
    asyncio.run(main())
