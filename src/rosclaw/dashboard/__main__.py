"""Entry point: python -m rosclaw.dashboard"""

from rosclaw.dashboard.launcher import serve_dashboard

if __name__ == "__main__":
    serve_dashboard(host="0.0.0.0", port=8765)
