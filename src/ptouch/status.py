# SPDX-FileCopyrightText: 2024-2026 Nicolai Buchwitz <nb@tipi-net.de>
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Printer status queries for Brother P-touch printers.

The raster command reference defines a fixed 32-byte status block that is
returned by the status request command (``ESC i S``) over USB and by the
P-touch Template status command (``^SR``).  On network printers the raw
printing port (TCP 9100) is a one-way pipe: Brother's own documentation
shows no status response in the network flow chart, and in practice the
printer never answers a status request on that socket.

Brother network printers expose the very same 32-byte block through SNMP
at the vendor OID ``1.3.6.1.4.1.2435.3.3.9.1.6.1.0``.  That is the
supported way to read status over the network and is what
:func:`query_status_snmp` uses.  If the vendor OID is unavailable, the
standard Printer MIB objects (media name and console display text) are used
as a fallback so that at least the loaded media width and the device
status can be reported.
"""

from __future__ import annotations

import random
import re
import socket
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Sequence

from .connection import PrinterConnectionError
from .printer import MediaType

#: Length of the status block returned by the printer, in bytes.
STATUS_RESPONSE_LENGTH = 32

#: Brother vendor OID returning the 32-byte status block over SNMP.
SNMP_VENDOR_STATUS_OID = "1.3.6.1.4.1.2435.3.3.9.1.6.1.0"

#: Standard Printer MIB objects used when the vendor OID is not available.
SNMP_HR_DEVICE_DESCR_OID = "1.3.6.1.2.1.25.3.2.1.3.1"
SNMP_PRINTER_SERIAL_OID = "1.3.6.1.2.1.43.5.1.1.17.1"
SNMP_INPUT_MEDIA_NAME_OID = "1.3.6.1.2.1.43.8.2.1.12.1.1"
SNMP_CONSOLE_DISPLAY_OID = "1.3.6.1.2.1.43.16.5.1.2.1.1"


class StatusType(IntEnum):
    """Status type reported in byte 18 of the status block."""

    REPLY_TO_STATUS_REQUEST = 0x00
    PRINTING_COMPLETED = 0x01
    ERROR_OCCURRED = 0x02
    TURNED_OFF = 0x04
    NOTIFICATION = 0x05
    PHASE_CHANGE = 0x06


class PhaseType(IntEnum):
    """Phase type reported in byte 19 of the status block."""

    EDITING_STATE = 0x00
    PRINTING_STATE = 0x01


#: Human readable names for the media types reported in byte 11.
MEDIA_TYPE_LABELS: dict[MediaType, str] = {
    MediaType.NO_MEDIA: "no media",
    MediaType.LAMINATED_TAPE: "laminated tape",
    MediaType.NONLAMINATED_TAPE: "non-laminated tape",
    MediaType.FABRIC_TAPE: "fabric tape",
    MediaType.HEATSHRINK_TUBE_21: "heat-shrink tube (2:1)",
    MediaType.HEATSHRINK_TUBE_31: "heat-shrink tube (3:1)",
    MediaType.FLE_TAPE: "FLe tape",
    MediaType.FLEXIBLE_ID_TAPE: "flexible ID tape",
    MediaType.INCOMPATIBLE_TAPE: "incompatible tape",
}

#: Error information 1 flags (byte 8).
ERROR_INFO_1: dict[int, str] = {
    0x01: "no media",
    0x02: "end of media",
    0x04: "cutter jam",
    0x08: "weak batteries",
    0x40: "high-voltage adapter",
}

#: Error information 2 flags (byte 9).
ERROR_INFO_2: dict[int, str] = {
    0x01: "wrong media",
    0x02: "expansion buffer full",
    0x04: "communication error",
    0x08: "communication buffer full",
    0x10: "cover open",
    0x20: "overheating",
    0x40: "black marking not detected",
    0x80: "system error",
}


class SnmpError(PrinterConnectionError):
    """Error while querying a printer over SNMP."""


@dataclass(frozen=True)
class PrinterStatus:
    """Status information reported by a Brother P-touch printer.

    Attributes
    ----------
    model : str or None
        Printer model name (e.g. ``"Brother PT-P900W"``), when known.
    serial : str or None
        Printer serial number, when known.
    media_width_mm : float or None
        Width of the loaded media in millimeters, or None when no media is
        loaded or the width is unknown.
    media_type_code : int or None
        Raw media type code from byte 11 of the status block.
    media_length_mm : int or None
        Media length in millimeters (0 for continuous tape).
    battery_level : int or None
        Raw battery level byte (byte 6).
    errors : tuple of str
        Decoded error messages. Empty when the printer reports no errors.
    error_info_1 : int or None
        Raw error information 1 byte (byte 8).
    error_info_2 : int or None
        Raw error information 2 byte (byte 9).
    display_status : str or None
        Device status text reported by the Printer MIB display object.
    status_type : int or None
        Raw status type byte (byte 18).
    phase_type : int or None
        Raw phase type byte (byte 19).
    phase_number : int or None
        Phase number (bytes 20-21, big endian).
    tape_color : int or None
        Raw tape colour code (byte 24).
    text_color : int or None
        Raw text colour code (byte 25).
    model_code : int or None
        Raw model code from byte 4 of the status block.
    dpi : int or None
        Printer resolution in DPI, when the model is known.
    max_px : int or None
        Maximum printable width in pixels, when the model is known.
    tape_px : int or None
        Printable height of the loaded media in pixels, when known.
    connected : bool
        Whether the printer answered the status request.
    message : str or None
        Additional human-readable status message, when one was set.
    raw : bytes
        The 32-byte status block, when one was returned.
    source : str
        Where the status came from: ``"vendor-status"`` (the full 32-byte
        block) or ``"printer-mib"`` (fallback using standard Printer MIB
        objects only).
    """

    model: str | None = None
    serial: str | None = None
    media_width_mm: float | None = None
    media_type_code: int | None = None
    media_length_mm: int | None = None
    battery_level: int | None = None
    errors: tuple[str, ...] = ()
    error_info_1: int | None = None
    error_info_2: int | None = None
    display_status: str | None = None
    status_type: int | None = None
    phase_type: int | None = None
    phase_number: int | None = None
    tape_color: int | None = None
    text_color: int | None = None
    model_code: int | None = None
    dpi: int | None = None
    max_px: int | None = None
    tape_px: int | None = None
    connected: bool = True
    message: str | None = None
    raw: bytes = b""
    source: str = "vendor-status"

    @property
    def media_type(self) -> MediaType | None:
        """Media type enum, or None when the code is unknown."""
        if self.media_type_code is None:
            return None
        try:
            return MediaType(self.media_type_code)
        except ValueError:
            return None

    @property
    def media_type_label(self) -> str:
        """Human readable media type."""
        media_type = self.media_type
        if media_type is None:
            if self.media_type_code is None:
                return "unknown"
            return f"unknown (0x{self.media_type_code:02X})"
        return MEDIA_TYPE_LABELS.get(media_type, media_type.name.lower())

    @property
    def model_name(self) -> str | None:
        """Short model name, e.g. ``"PT-P900W"`` instead of the description."""
        if not self.model:
            return None
        match = re.search(r"PT-?[A-Z0-9]+", self.model.upper())
        if match:
            return match.group(0)
        return self.model.strip()

    @property
    def error_code(self) -> int | None:
        """Combined 16-bit error code: error information 1 in the low byte."""
        if self.error_info_1 is None and self.error_info_2 is None:
            return None
        return (self.error_info_1 or 0) | ((self.error_info_2 or 0) << 8)

    @property
    def ready(self) -> bool:
        """Whether the printer is connected, has media and reports no error."""
        if not self.connected or self.is_error:
            return False
        if self.media_type is MediaType.NO_MEDIA:
            return False
        if self.media_width_mm is None or self.media_width_mm <= 0:
            return False
        return True

    @property
    def device_status(self) -> str:
        """Device status, e.g. ``"READY"`` or ``"ERROR"``."""
        if not self.connected:
            return "NOT CONNECTED"
        if self.errors:
            return "ERROR"
        if self.display_status:
            text = self.display_status.strip()
            if text:
                return text
        if self.status_type == StatusType.ERROR_OCCURRED:
            return "ERROR"
        if self.status_type in (
            StatusType.REPLY_TO_STATUS_REQUEST,
            StatusType.PRINTING_COMPLETED,
        ):
            return "READY"
        return "UNKNOWN"

    @property
    def media_status(self) -> str:
        """Media status, e.g. ``"Not Empty"`` or ``"Empty"``."""
        if self.media_type is MediaType.NO_MEDIA:
            return "Empty"
        if self.media_width_mm is None:
            return "Unknown"
        if self.media_width_mm <= 0:
            return "Empty"
        return "Not Empty"

    @property
    def is_error(self) -> bool:
        """Whether the printer currently reports an error."""
        if self.errors:
            return True
        if self.status_type == StatusType.ERROR_OCCURRED:
            return True
        return bool(self.display_status and self.display_status.strip().upper().startswith("ERROR"))


def parse_status_response(
    data: bytes,
    *,
    model: str | None = None,
    serial: str | None = None,
) -> PrinterStatus:
    """Parse a 32-byte Brother status block.

    Parameters
    ----------
    data : bytes
        Status bytes returned by ``ESC i S``, ``^SR`` or the vendor SNMP
        OID.  At least :data:`STATUS_RESPONSE_LENGTH` bytes are required;
        extra bytes are ignored.
    model : str or None, optional
        Printer model name to attach to the result.
    serial : str or None, optional
        Printer serial number to attach to the result.

    Returns
    -------
    PrinterStatus
        Parsed status information.

    Raises
    ------
    ValueError
        If the data is too short or does not start with the Brother status
        header (``80 20 42``).
    """
    if len(data) < STATUS_RESPONSE_LENGTH:
        raise ValueError(
            f"Status response must be at least {STATUS_RESPONSE_LENGTH} bytes, got {len(data)}"
        )

    data = data[:STATUS_RESPONSE_LENGTH]
    if data[0] != 0x80 or data[1] != 0x20 or data[2] != 0x42:
        raise ValueError(f"Invalid status header: {data[:3].hex(' ')}")

    errors = tuple(name for bit, name in ERROR_INFO_1.items() if data[8] & bit) + tuple(
        name for bit, name in ERROR_INFO_2.items() if data[9] & bit
    )

    width = data[10]
    media_width_mm = float(width) if 0 < width < 0xFF else None

    return PrinterStatus(
        model=model,
        serial=serial,
        media_width_mm=media_width_mm,
        media_type_code=data[11],
        media_length_mm=data[17],
        battery_level=data[6],
        errors=errors,
        error_info_1=data[8],
        error_info_2=data[9],
        status_type=data[18],
        phase_type=data[19],
        phase_number=(data[20] << 8) | data[21],
        tape_color=data[24],
        text_color=data[25],
        model_code=data[4],
        raw=data,
        source="vendor-status",
    )


def parse_media_name(name: str | None) -> float | None:
    """Extract the media width from a Printer MIB media name.

    Parameters
    ----------
    name : str or None
        Media name string, e.g. ``'24mm(0.94")'``.

    Returns
    -------
    float or None
        Width in millimeters, or None if no width could be parsed.
    """
    if not name:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mm", name, re.IGNORECASE)
    return float(match.group(1)) if match else None


def query_status_snmp(
    host: str,
    *,
    community: str = "public",
    version: str = "2c",
    port: int = 161,
    timeout: float = 3.0,
) -> PrinterStatus:
    """Query a network printer's status over SNMP.

    Parameters
    ----------
    host : str
        Hostname or IP address of the printer.
    community : str, default "public"
        SNMP read community.
    version : str, default "2c"
        SNMP version, either ``"1"`` or ``"2c"``.
    port : int, default 161
        SNMP UDP port.
    timeout : float, default 3.0
        Timeout per SNMP request in seconds.

    Returns
    -------
    PrinterStatus
        Parsed status information. When the vendor status OID is not
        available, the result is built from the standard Printer MIB and
        ``source`` is ``"printer-mib"``.

    Raises
    ------
    SnmpError
        If the printer cannot be reached or exposes no status information.
    """
    varbinds = snmp_get(
        host,
        [
            SNMP_VENDOR_STATUS_OID,
            SNMP_HR_DEVICE_DESCR_OID,
            SNMP_PRINTER_SERIAL_OID,
            SNMP_INPUT_MEDIA_NAME_OID,
            SNMP_CONSOLE_DISPLAY_OID,
        ],
        community=community,
        version=version,
        port=port,
        timeout=timeout,
    )

    model = _decode_snmp_string(varbinds.get(SNMP_HR_DEVICE_DESCR_OID))
    serial = _decode_snmp_string(varbinds.get(SNMP_PRINTER_SERIAL_OID))
    raw = varbinds.get(SNMP_VENDOR_STATUS_OID)

    if isinstance(raw, bytes) and len(raw) >= STATUS_RESPONSE_LENGTH:
        status = parse_status_response(raw, model=model, serial=serial)
        return _enrich(status)

    media_name = _decode_snmp_string(varbinds.get(SNMP_INPUT_MEDIA_NAME_OID))
    display = _decode_snmp_string(varbinds.get(SNMP_CONSOLE_DISPLAY_OID))
    if media_name is None and display is None:
        raise SnmpError(
            f"Printer at {host} does not expose status information over SNMP "
            f"(vendor OID {SNMP_VENDOR_STATUS_OID} and Printer MIB objects are missing)"
        )

    return _enrich(
        PrinterStatus(
            model=model,
            serial=serial,
            media_width_mm=parse_media_name(media_name),
            display_status=display,
            source="printer-mib",
        )
    )


def _enrich(status: PrinterStatus) -> PrinterStatus:
    """Add model-specific fields without a circular module import."""
    from .detect import enrich_status

    return enrich_status(status)


def snmp_get(
    host: str,
    oids: Sequence[str],
    *,
    community: str = "public",
    version: str = "2c",
    port: int = 161,
    timeout: float = 3.0,
    retries: int = 2,
) -> dict[str, Any]:
    """Perform an SNMP GET for a list of OIDs.

    This is a deliberately small SNMP v1/v2c client so that status queries
    do not pull in an external SNMP dependency.  It only supports the GET
    operation needed by :func:`query_status_snmp`.

    Parameters
    ----------
    host : str
        Hostname or IP address of the device.
    oids : sequence of str
        OIDs to request.  Leading dots are ignored.
    community : str, default "public"
        SNMP read community.
    version : str, default "2c"
        SNMP version, either ``"1"`` or ``"2c"``.
    port : int, default 161
        SNMP UDP port.
    timeout : float, default 3.0
        Timeout per request in seconds.
    retries : int, default 2
        Number of retries after the first attempt.

    Returns
    -------
    dict
        Mapping of requested OID to value.  Values are ``bytes`` for octet
        strings, ``int`` for integers and counters, ``str`` for OIDs, and
        ``None`` for missing objects.  Missing OIDs are present with a
        value of ``None``.

    Raises
    ------
    ValueError
        If *version* is not ``"1"`` or ``"2c"``, or an OID is invalid.
    SnmpError
        If the device does not respond or returns an SNMP error.
    """
    if version not in ("1", "2c"):
        raise ValueError(f"Unsupported SNMP version: {version!r} (expected '1' or '2c')")

    normalized = [oid.lstrip(".") for oid in oids]
    payload, request_id = _build_get_request(
        normalized, community=community, version=0 if version == "1" else 1
    )
    response = _udp_exchange(payload, host, port, timeout, retries)
    error_status, error_index, varbinds = _parse_snmp_response(response, request_id)

    result: dict[str, Any] = dict.fromkeys(normalized)
    if error_status != 0:
        # SNMPv1 signals a missing object with noSuchName (status 2) and no
        # varbinds; treat it like the SNMPv2c noSuchObject exception.
        if error_status == 2:
            return result
        raise SnmpError(f"SNMP error from {host}: status {error_status}, index {error_index}")

    for oid, tag, value in varbinds:
        result[oid] = None if tag in (0x80, 0x81, 0x82) else value
    return result


def _decode_snmp_string(value: Any) -> str | None:
    """Decode an octet string SNMP value to a stripped string."""
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace").strip()
    if isinstance(value, str):
        return value.strip()
    return None


def _build_get_request(
    oids: Sequence[str],
    *,
    community: str,
    version: int,
) -> tuple[bytes, int]:
    """Build an SNMP GET request and return it with its request id."""
    request_id = random.SystemRandom().randint(1, 0x7FFFFFFF)
    varbinds = b"".join(_tlv(0x30, _encode_oid(oid) + b"\x05\x00") for oid in oids)
    pdu = _tlv(
        0xA0,
        _encode_integer(request_id)
        + _encode_integer(0)
        + _encode_integer(0)
        + _tlv(0x30, varbinds),
    )
    message = _tlv(
        0x30,
        _encode_integer(version) + _tlv(0x04, community.encode("utf-8")) + pdu,
    )
    return message, request_id


def _udp_exchange(payload: bytes, host: str, port: int, timeout: float, retries: int) -> bytes:
    """Send a datagram and return the first response, retrying on timeout."""
    last_error: Exception | None = None
    for _ in range(retries + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        try:
            sock.sendto(payload, (host, port))
            data, _ = sock.recvfrom(65535)
            return data
        except socket.timeout as e:
            last_error = e
        except OSError as e:
            raise SnmpError(
                f"SNMP request to {host}:{port} failed: {e}",
                original_error=e,
            ) from e
        finally:
            sock.close()
    raise SnmpError(
        f"No SNMP response from {host}:{port} after {retries + 1} attempt(s)",
        original_error=last_error,
    )


def _parse_snmp_response(
    data: bytes, request_id: int
) -> tuple[int, int, list[tuple[str, int, Any]]]:
    """Parse an SNMP GetResponse and return error status, index and varbinds."""
    tag, message, _ = _read_tlv(data, 0)
    if tag != 0x30:
        raise SnmpError(f"Invalid SNMP message tag 0x{tag:02X}")

    pos = 0
    _, _, pos = _read_tlv(message, pos)  # version
    _, _, pos = _read_tlv(message, pos)  # community
    tag, pdu, pos = _read_tlv(message, pos)
    if tag != 0xA2:
        raise SnmpError(f"Unexpected SNMP PDU type 0x{tag:02X} (expected GetResponse)")

    p = 0
    _, raw_request_id, p = _read_tlv(pdu, p)
    _, raw_error_status, p = _read_tlv(pdu, p)
    _, raw_error_index, p = _read_tlv(pdu, p)
    if _decode_integer(raw_request_id) != request_id:
        raise SnmpError("SNMP response request-id does not match the request")

    varbinds: list[tuple[str, int, Any]] = []
    if p < len(pdu):
        _, raw_varbinds, p = _read_tlv(pdu, p)
        pos = 0
        while pos < len(raw_varbinds):
            _, raw_varbind, pos = _read_tlv(raw_varbinds, pos)
            q = 0
            _, raw_oid, q = _read_tlv(raw_varbind, q)
            value_tag, raw_value, q = _read_tlv(raw_varbind, q)
            varbinds.append(
                (
                    _decode_oid(raw_oid),
                    value_tag,
                    _decode_value(value_tag, raw_value),
                )
            )

    return _decode_integer(raw_error_status), _decode_integer(raw_error_index), varbinds


def _decode_value(tag: int, payload: bytes) -> Any:
    """Decode an SNMP varbind value based on its BER tag."""
    if tag == 0x02:
        return _decode_integer(payload)
    if tag == 0x04:
        return payload
    if tag == 0x05:
        return None
    if tag == 0x06:
        return _decode_oid(payload)
    if tag in (0x41, 0x42, 0x43):
        return int.from_bytes(payload, "big", signed=False)
    if tag in (0x80, 0x81, 0x82):
        return None
    return payload


def _tlv(tag: int, payload: bytes) -> bytes:
    """Encode a BER tag-length-value triplet."""
    return bytes([tag]) + _encode_length(len(payload)) + payload


def _encode_length(length: int) -> bytes:
    """Encode a BER length."""
    if length < 0x80:
        return bytes([length])
    raw = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(raw)]) + raw


def _encode_integer(value: int) -> bytes:
    """Encode a BER INTEGER."""
    length = max(1, (value.bit_length() + 8) // 8)
    return _tlv(0x02, value.to_bytes(length, "big", signed=True))


def _encode_oid(oid: str) -> bytes:
    """Encode a dotted OID as a BER OBJECT IDENTIFIER."""
    parts = oid.lstrip(".").split(".")
    if len(parts) < 2:
        raise ValueError(f"OID must have at least two arcs: {oid!r}")
    try:
        arcs = [int(part) for part in parts]
    except ValueError as e:
        raise ValueError(f"Invalid OID: {oid!r}") from e
    if arcs[0] not in (0, 1, 2) or arcs[1] >= 40:
        raise ValueError(f"Invalid OID prefix: {oid!r}")

    payload = bytearray([40 * arcs[0] + arcs[1]])
    for arc in arcs[2:]:
        if arc < 0:
            raise ValueError(f"Negative OID arc in {oid!r}")
        chunk = [arc & 0x7F]
        arc >>= 7
        while arc:
            chunk.append((arc & 0x7F) | 0x80)
            arc >>= 7
        payload.extend(reversed(chunk))
    return _tlv(0x06, bytes(payload))


def _decode_oid(payload: bytes) -> str:
    """Decode a BER OBJECT IDENTIFIER to dotted form."""
    if not payload:
        return ""
    first = payload[0]
    arcs = [first // 40, first % 40]
    value = 0
    for byte in payload[1:]:
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            arcs.append(value)
            value = 0
    return ".".join(str(arc) for arc in arcs)


def _decode_integer(payload: bytes) -> int:
    """Decode a BER INTEGER."""
    return int.from_bytes(payload, "big", signed=True)


def _read_tlv(data: bytes, offset: int) -> tuple[int, bytes, int]:
    """Read a BER tag-length-value triplet at *offset*."""
    if offset >= len(data):
        raise SnmpError("Truncated SNMP message")
    tag = data[offset]
    offset += 1
    if offset >= len(data):
        raise SnmpError("Truncated SNMP length")
    first = data[offset]
    offset += 1
    if first < 0x80:
        length = first
    elif first == 0x80:
        raise SnmpError("Indefinite BER lengths are not supported")
    else:
        count = first & 0x7F
        if count > 4 or offset + count > len(data):
            raise SnmpError("Invalid BER length")
        length = int.from_bytes(data[offset : offset + count], "big")
        offset += count
    end = offset + length
    if end > len(data):
        raise SnmpError("Truncated SNMP value")
    return tag, data[offset:end], end
