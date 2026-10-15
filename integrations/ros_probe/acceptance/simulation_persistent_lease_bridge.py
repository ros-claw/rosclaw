"""SDK-only ROS bridge shim: retain one real client for the declared SIM lease.

No response cache, extra ROS node, publisher, lease extension or new motion
route. Other services use the unmodified upstream implementation.
"""

import argparse
import hashlib
import json
import os
import runpy
import sys
import threading
import time
import weakref
from pathlib import Path

UPSTREAM_SERVICES_SHA256 = "47fd035f751a90464a40a8dfebcc820d83ea5fdc33d88f6b46225d59dbd55ddd"


class PersistentLeaseService:
    def __init__(self, service, upstream, *, create_client, prepare_request, extract_values, audit):
        self.service = service
        self.upstream = upstream
        self.create_client = create_client
        self.prepare_request = prepare_request
        self.extract_values = extract_values
        self.audit = audit
        self.clients = weakref.WeakKeyDictionary()
        self.lock = threading.Lock()

    def __call__(
        self, node, service, args=None, server_ready_timeout=1.0, server_response_timeout=5.0
    ):
        resolved = node.resolve_service_name(service)
        if resolved != self.service:
            return self.upstream(node, service, args, server_ready_timeout, server_response_timeout)
        if type(args) is not dict or set(args) != {"data"} or type(args["data"]) is not bool:
            raise ValueError("declared SIM lease accepts only actual SetBool requests")
        if dict(node.get_service_names_and_types()).get(resolved) != ["std_srvs/srv/SetBool"]:
            raise ValueError("declared SIM lease service type unavailable or changed")
        with self.lock:
            client = self.clients.get(node)
            if client is None:
                client = self.create_client(node, resolved)
                self.clients[node] = client
                self.audit({"event": "PERSISTENT_LEASE_CLIENT_CREATED", "service": resolved})
        if not client.wait_for_service(server_ready_timeout):
            raise ValueError("declared SIM lease service unavailable")
        started = time.monotonic_ns()
        future = client.call_async(self.prepare_request(args))
        complete = threading.Event()
        future.add_done_callback(lambda _: complete.set())
        if not complete.wait(server_response_timeout if server_response_timeout > 0 else None):
            future.cancel()
            self.audit({"event": "LEASE_SERVICE_RESPONSE_TIMEOUT", "started_ns": started})
            raise RuntimeError("Timeout exceeded while waiting for actual lease service response")
        result = future.result()
        if result is None:
            raise RuntimeError("actual lease service exception: " + str(future.exception()))
        response = self.extract_values(result)
        self.audit(
            {
                "event": "ACTUAL_LEASE_SERVICE_RESPONSE",
                "started_ns": started,
                "completed_ns": time.monotonic_ns(),
                "requested_enabled": args["data"],
                "success": response.get("success"),
            }
        )
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lease-service", required=True)
    args, ros_args = parser.parse_known_args()
    if (
        not Path("/.dockerenv").is_file()
        or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
        or not args.lease_service.startswith("/")
        or ".." in args.lease_service
        or args.lease_service.endswith("/")
    ):
        raise ValueError("isolated SDK declared SIM lease service required")
    settings = json.loads(Path("/settings.json").read_text())
    sources = json.loads(Path("/assets/observer-sources.json").read_text())
    if (
        settings.get("generic_source_observer") is not True
        or "governed_sink" not in settings
        or settings["run_id"] != sources["run_id"]
        or args.lease_service != sources["topics"]["lease"]
    ):
        raise ValueError("closed prepared governed SDK lease context required")
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rosbridge_library.internal import services
    from rosbridge_library.internal.executor_helpers import run_on_executor
    from std_srvs.srv import SetBool

    if hashlib.sha256(Path(services.__file__).read_bytes()).hexdigest() != UPSTREAM_SERVICES_SHA256:
        raise ValueError("pinned SDK upstream ROS bridge service implementation changed")
    audit_lock = threading.Lock()
    audit_path = Path("/evidence/persistent-lease-client-original.jsonl")
    audit_path.touch(exist_ok=False)
    count = 0

    def audit(value):
        nonlocal count
        with audit_lock:
            if count >= 2000:
                raise ValueError("bounded persistent lease client audit exhausted")
            count += 1
            with audit_path.open("a") as stream:
                stream.write(json.dumps({"monotonic_ns": time.monotonic_ns(), **value}) + "\n")

    def create_client(node, service):
        return run_on_executor(
            node,
            lambda: node.create_client(SetBool, service, callback_group=ReentrantCallbackGroup()),
        )

    def prepare_request(values):
        request = SetBool.Request()
        request.data = values["data"]
        return request

    services.call_service = PersistentLeaseService(
        args.lease_service,
        services.call_service,
        create_client=create_client,
        prepare_request=prepare_request,
        extract_values=services.extract_values,
        audit=audit,
    )
    audit(
        {
            "event": "PINNED_UPSTREAM_SIM_LEASE_CLIENT_REUSE_INSTALLED",
            "upstream_services_sha256": UPSTREAM_SERVICES_SHA256,
            "service": args.lease_service,
            "responses_cached": False,
        }
    )
    sys.argv = ["rosbridge_websocket", *ros_args]
    runpy.run_path(
        "/opt/ros/jazzy/lib/rosbridge_server/rosbridge_websocket.py", run_name="__main__"
    )


if __name__ == "__main__":
    main()
