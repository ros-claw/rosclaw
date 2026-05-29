#!/usr/bin/env python3
"""Start Dashboard server for v1.0 acceptance."""
import asyncio
from rosclaw.dashboard.server import DashboardServer

server = DashboardServer(host='0.0.0.0', port=8080)
asyncio.run(server.start())
print('Dashboard started on http://localhost:8080')
