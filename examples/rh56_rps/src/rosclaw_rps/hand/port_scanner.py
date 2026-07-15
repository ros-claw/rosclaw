"""Read-only RH56 hand port discovery.

Scans available USB/serial ports and probes the configured Modbus slave ids
(1 for left hand, 2 for right hand by default).  This lets the demo adapt
dynamically when /dev/ttyUSB0 and /dev/ttyUSB1 are swapped or when hands are
plugged into different ports.
"""
from __future__ import annotations

import glob
import logging
import termios
import time
from typing import Iterable

import serial

from rosclaw_rh56.protocol.inspire_protocol import (
    build_read_holding_registers,
    parse_read_response,
    verify_modbus_response,
)

logger = logging.getLogger("rosclaw_rps.port_scanner")

# RH56 default holding register that contains the configured slave id.
_HAND_ID_REGISTER = 0x03E8
_DEFAULT_BAUDRATE = 115200
_DEFAULT_TIMEOUT_S = 0.3

# Re-opening CH340 devices repeatedly in quick succession can trigger driver
# -110 errors.  We therefore keep discovered ports *open* and hand the open
# serial object off to the caller instead of closing and reopening.
_cache: dict[int, str] | None = None
_cache_time: float = 0.0
_open_serials: dict[str, serial.Serial] = {}


def _configure_serial_for_safe_reopen(ser: serial.Serial) -> None:
    """Keep DTR asserted when the port is closed.

    The default termios ``HUPCL`` flag tells the driver to drop DTR on the last
    close.  Some CH340 USB/RS485 adapters interpret that drop as a reset and
    subsequently fail to reopen with ``Input/output error`` or kernel ``-110``.
    Clearing ``HUPCL`` lets DTR stay high across close/reopen cycles.
    """
    if ser.fd is None:
        return
    try:
        attrs = termios.tcgetattr(ser.fd)
        attrs[2] &= ~termios.HUPCL
        termios.tcsetattr(ser.fd, termios.TCSANOW, attrs)
    except Exception:
        pass


def _probe_slave(ser: serial.Serial, slave_id: int) -> bool:
    """Return True if the open serial device answers to *slave_id* on HAND_ID."""
    request = build_read_holding_registers(slave_id, _HAND_ID_REGISTER, quantity=1)
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        ser.write(request)
        time.sleep(0.05)
        response = ser.read(64)
    except Exception as exc:
        logger.debug("Probe slave=%s on %s failed: %s", slave_id, ser.port, exc)
        return False

    if not verify_modbus_response(response, slave_id, function_code=0x03):
        return False
    values = parse_read_response(response)
    return bool(values) and values[0] == slave_id


def _take_open_serial(port: str) -> serial.Serial | None:
    """Return an already-open serial object for *port* if we have one."""
    ser = _open_serials.get(port)
    if ser is not None and ser.is_open:
        return ser
    return None


def discover_hands(
    port_glob: str = "/dev/ttyUSB*",
    slave_ids: Iterable[int] = (1, 2),
    baudrate: int = _DEFAULT_BAUDRATE,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    retries: int = 2,
    use_cache: bool = True,
    cache_ttl_s: float = 300.0,
) -> dict[int, str]:
    """Discover which serial port each RH56 hand is attached to.

    Each candidate port is opened at most once and kept open if a matching
    slave id is found.  The open ``serial.Serial`` objects can be retrieved
    with :func:`get_open_serial` and passed directly to the hand controller,
    avoiding the close/reopen cycle that destabilizes some CH340 adapters.

    Args:
        port_glob: Glob pattern for candidate serial devices.
        slave_ids: Modbus slave ids to look for (default left=1, right=2).
        baudrate: Serial baudrate.
        timeout_s: Read timeout for each probe.
        retries: Number of attempts per port if open/probe fails.
        use_cache: Whether to return a recent cached mapping if available.
        cache_ttl_s: How long a cached mapping remains valid.

    Returns:
        Mapping from slave_id to port path.
    """
    global _cache, _cache_time

    if use_cache and _cache is not None:
        age = time.time() - _cache_time
        if age < cache_ttl_s:
            logger.debug("Using cached hand mapping (age=%.2fs): %s", age, _cache)
            return _cache.copy()

    ports = sorted(glob.glob(port_glob))
    if not ports:
        raise RuntimeError(f"No serial ports matched by {port_glob}")

    ids = list(slave_ids)
    mapping: dict[int, str] = {}

    for port in ports:
        found_on_port: set[int] = set()
        for attempt in range(retries):
            ser: serial.Serial | None = None
            try:
                # Reuse an already-open port from an earlier discovery pass.
                ser = _take_open_serial(port)
                if ser is None:
                    ser = serial.Serial(port, baudrate=baudrate, timeout=timeout_s)
                    # Give the CH340 a moment after open before sending traffic.
                    time.sleep(0.05)
                    _configure_serial_for_safe_reopen(ser)
                for slave_id in ids:
                    if slave_id in mapping:
                        continue
                    if _probe_slave(ser, slave_id):
                        mapping[slave_id] = port
                        found_on_port.add(slave_id)
                        # Keep the port open and remember it for the controller.
                        _open_serials[port] = ser
                        logger.info(
                            "Discovered RH56 slave id %s on %s", slave_id, port
                        )
                    time.sleep(0.02)
                break
            except Exception as exc:
                logger.debug(
                    "Port %s open/probe failed (attempt %s): %s", port, attempt, exc
                )
                time.sleep(0.05)
            finally:
                # Only close ports where we did not find a hand.
                if ser is not None and port not in _open_serials:
                    try:
                        ser.close()
                    except Exception:
                        pass

        if len(found_on_port) > 1:
            raise ValueError(
                f"Multiple slave ids {sorted(found_on_port)} responded on {port}. "
                "Check that each hand uses a unique id."
            )

    _cache = mapping.copy()
    _cache_time = time.time()
    return mapping


def get_open_serial(port: str) -> serial.Serial | None:
    """Return an already-open ``serial.Serial`` for *port* if available.

    This is used by the hand controller to take over a port that was kept open
    during discovery, avoiding a destructive close/reopen cycle.
    """
    return _take_open_serial(port)


def clear_port_cache() -> None:
    """Clear the in-process discovery cache and close any ports we still hold."""
    global _cache, _cache_time
    _cache = None
    _cache_time = 0.0
    for ser in list(_open_serials.values()):
        try:
            if ser.is_open:
                try:
                    ser.flush()
                except Exception:
                    pass
                time.sleep(0.02)
                ser.close()
        except Exception:
            pass
    _open_serials.clear()


def resolve_port(
    device_id: int,
    preferred_port: str,
    port_glob: str = "/dev/ttyUSB*",
    slave_ids: Iterable[int] = (1, 2),
) -> str:
    """Resolve a hand port, supporting ``auto`` for dynamic discovery.

    If *preferred_port* is ``"auto"`` the function scans for the configured
    *slave_ids* at once and returns the port matching *device_id*.  The open
    serial object can be retrieved with :func:`get_open_serial` and passed to
    the controller so the port is not closed and reopened.
    """
    if preferred_port.lower() != "auto":
        return preferred_port
    mapping = discover_hands(port_glob=port_glob, slave_ids=slave_ids)
    if device_id not in mapping:
        raise RuntimeError(
            f"Could not find RH56 hand with slave id {device_id} "
            f"on ports matching {port_glob}"
        )
    return mapping[device_id]
