"""Original inclusive header-age contracts using original integer ROS times.

Never reconstruct original nanoseconds from a float. Older producers without
integer fields keep their original float contract; incomplete or inconsistent
integer facts fail rather than silently falling back. No threshold is widened.
"""

import math
from decimal import Decimal


def stamp_ns(stamp):
    if (
        type(stamp.sec) is not int
        or type(stamp.nanosec) is not int
        or not 0 <= stamp.nanosec < 1_000_000_000
    ):
        raise ValueError("Original canonical integer ROS stamp required")
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def integer_age_ok(native_ns, header_ns, future_s, max_age_s):
    if type(native_ns) is not int or type(header_ns) is not int:
        raise ValueError("Original integer native/header nanoseconds required")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (future_s, max_age_s)
    ):
        raise ValueError("Original finite nonnegative header limits required")
    age = native_ns - header_ns
    return -Decimal(str(future_s)) * 1_000_000_000 <= age <= Decimal(str(max_age_s)) * 1_000_000_000


def header_age_ok(native_s, header_s, future_s, max_age_s, *, native_ns=None, header_ns=None):
    if native_ns is None and header_ns is None:
        return (
            header_s is not None
            and math.isfinite(header_s)
            and -future_s <= native_s - header_s <= max_age_s
        )
    if type(native_ns) is not int or type(header_ns) is not int:
        raise ValueError("Incomplete original integer time pair")
    if native_s != native_ns / 1e9 or header_s != header_ns / 1e9:
        raise ValueError("Original integer time and seconds projection differ")
    return integer_age_ok(native_ns, header_ns, future_s, max_age_s)
