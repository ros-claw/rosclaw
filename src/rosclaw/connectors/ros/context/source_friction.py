"""Pure bounded source friction facts shared by diagnosis and SIM preparation."""

import math
import xml.etree.ElementTree as ET

_FRAME_URIS = {"http://ignitionrobotics.org/schema", "http://gazebosim.org/schemas"}


def parse_source_xml(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 4_000_000:
        raise ValueError("bounded original UTF-8 XML bytes required")
    return ET.fromstring(raw.decode("utf-8"))


def source_friction_fact(link, ode, names):
    direction = ode.find("fdir1")
    if direction is None:
        return None
    attrs = list(direction.attrib.items())
    if len(attrs) != 1:
        raise ValueError("one explicit namespaced original friction frame required")
    key, frame = attrs[0]
    if not key.startswith("{") or not key.endswith("}expressed_in"):
        raise ValueError("unsupported friction frame attribute")
    uri = key[1:].split("}", 1)[0]
    if uri not in _FRAME_URIS or frame not in names:
        raise ValueError("unresolved original friction direction frame")
    text = (direction.text or "").split()
    if len(text) != 3:
        raise ValueError("three original friction direction components required")
    values = [float(x) for x in text]
    coefficients = [float(ode.findtext(name, "nan")) for name in ["mu", "mu2"]]
    if (
        not all(math.isfinite(x) for x in values + coefficients)
        or not any(x != 0 for x in values)
        or any(x < 0 for x in coefficients)
    ):
        raise ValueError("finite original direction and nonnegative friction required")
    return (link, uri, frame, tuple(values), tuple(coefficients))
