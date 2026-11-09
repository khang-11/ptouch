# SPDX-FileCopyrightText: 2024-2026 Nicolai Buchwitz <nb@tipi-net.de>
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Printer and media detection helpers.

This module maps the model name reported by the printer (or the USB product
ID) to a printer class, and the media width/type reported in the status block
to a tape class.  It also implements LAN discovery over SNMP and USB device
enumeration for the command line interface.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any

from .printer import MediaType
from .printers import (
    PTE550W,
    PTP710BT,
    PTP750W,
    PTP900,
    PTP900W,
    PTP910BT,
    PTP950NW,
)
from .status import (
    SNMP_HR_DEVICE_DESCR_OID,
    SNMP_PRINTER_SERIAL_OID,
    SNMP_VENDOR_STATUS_OID,
    PrinterStatus,
    parse_status_response,
    snmp_get,
)
from .tape import (
    HeatShrinkTube3_1_5_2mm,
    HeatShrinkTube3_1_9_0mm,
    HeatShrinkTube3_1_11_2mm,
    HeatShrinkTube3_1_21_0mm,
    HeatShrinkTube3_1_31_0mm,
    HeatShrinkTube5_8mm,
    HeatShrinkTube8_8mm,
    HeatShrinkTube11_7mm,
    HeatShrinkTube17_7mm,
    HeatShrinkTube23_6mm,
    Tape,
    Tape3_5mm,
    Tape6mm,
    Tape9mm,
    Tape12mm,
    Tape18mm,
    Tape24mm,
    Tape36mm,
)

#: Mapping of tape width (mm) to tape classes.
TAPE_WIDTHS: dict[float, type[Tape]] = {
    3.5: Tape3_5mm,
    6: Tape6mm,
    9: Tape9mm,
    12: Tape12mm,
    18: Tape18mm,
    24: Tape24mm,
    36: Tape36mm,
}

#: Mapping of heat shrink tube diameter (mm) to tube classes.
TUBE_WIDTHS: dict[float, type[Tape]] = {
    5.8: HeatShrinkTube5_8mm,
    8.8: HeatShrinkTube8_8mm,
    11.7: HeatShrinkTube11_7mm,
    17.7: HeatShrinkTube17_7mm,
    23.6: HeatShrinkTube23_6mm,
    5.2: HeatShrinkTube3_1_5_2mm,
    9.0: HeatShrinkTube3_1_9_0mm,
    11.2: HeatShrinkTube3_1_11_2mm,
    21.0: HeatShrinkTube3_1_21_0mm,
    31.0: HeatShrinkTube3_1_31_0mm,
}

#: Mapping of short printer names to printer classes.
PRINTER_TYPES: dict[str, type] = {
    "E550W": PTE550W,
    "P710BT": PTP710BT,
    "P750W": PTP750W,
    "P900": PTP900,
    "P900W": PTP900W,
    "P910BT": PTP910BT,
    "P950NW": PTP950NW,
}

PRINTER_CLASSES = list(PRINTER_TYPES.values())

PRINTERS_BY_MODEL: dict[str, type] = {cls.MODEL_NAME: cls for cls in PRINTER_CLASSES}
PRINTERS_BY_MODEL.update(PRINTER_TYPES)

PRINTERS_BY_USB_PRODUCT_ID: dict[int, type] = {cls.USB_PRODUCT_ID: cls for cls in PRINTER_CLASSES}

#: Media type codes that mean a heat shrink tube.
_HEATSHRINK_MEDIA = {MediaType.HEATSHRINK_TUBE_21.value, MediaType.HEATSHRINK_TUBE_31.value}


def normalize_model_name(model: str | None) -> str | None:
    """Normalize a printer model description to ``PT-...`` or a short name.

    Parameters
    ----------
    model : str or None
        Model description such as ``"Brother PT-P900W"``, ``"PT-P900W"`` or
        ``"P900W"``.

    Returns
    -------
    str or None
        Normalized model name, or None when *model* is empty.
    """
    if not model:
        return None
    text = model.strip().upper()
    text = text.replace("BROTHER", "").strip()
    match = re.search(r"PT-?([A-Z0-9]+)", text)
    if match:
        return f"PT-{match.group(1)}"
    return text or None


def printer_class_for_model(model: str | None) -> type | None:
    """Return the printer class for a model description, if known."""
    name = normalize_model_name(model)
    if name is None:
        return None
    return PRINTERS_BY_MODEL.get(name)


def printer_class_for_usb_product_id(product_id: int) -> type | None:
    """Return the printer class for a Brother USB product ID, if known."""
    return PRINTERS_BY_USB_PRODUCT_ID.get(product_id)


def resolve_media_class(
    width_mm: float | None,
    media_type_code: int | None = None,
) -> type[Tape] | None:
    """Return the tape class for a media width and media type code.

    Parameters
    ----------
    width_mm : float or None
        Media width reported by the printer.
    media_type_code : int or None
        Raw media type code from the status block.

    Returns
    -------
    type[Tape] or None
        Matching tape or heat shrink tube class, or None when unknown.
    """
    if width_mm is None or width_mm <= 0:
        return None
    width = round(float(width_mm), 1)
    if media_type_code in _HEATSHRINK_MEDIA:
        return TUBE_WIDTHS.get(width)
    return TAPE_WIDTHS.get(width)


def max_pixel_width(printer_class: type) -> int | None:
    """Maximum printable width in pixels across all supported media."""
    if not printer_class.PIN_CONFIGS:
        return None
    return max(config.print_pins for config in printer_class.PIN_CONFIGS.values())


def tape_pixel_height(printer_class: type, tape_class: type[Tape] | None) -> int | None:
    """Printable height in pixels for a tape on a printer, if supported."""
    if tape_class is None:
        return None
    config = printer_class.PIN_CONFIGS.get(tape_class)
    return config.print_pins if config is not None else None


def enrich_status(
    status: PrinterStatus,
    printer_class: type | None = None,
) -> PrinterStatus:
    """Add model-specific fields (DPI, tape and printer pixel widths) to status.

    Parameters
    ----------
    status : PrinterStatus
        Status returned by the printer.
    printer_class : type or None, optional
        Printer class to use instead of resolving the model name.

    Returns
    -------
    PrinterStatus
        Status with ``dpi``, ``max_px`` and ``tape_px`` filled in when the
        printer and media are known.
    """
    printer_class = printer_class or printer_class_for_model(status.model)
    if printer_class is None:
        return status

    tape_class = resolve_media_class(status.media_width_mm, status.media_type_code)
    return replace(
        status,
        model=status.model or printer_class.MODEL_NAME,
        dpi=printer_class.RESOLUTION_DPI,
        max_px=max_pixel_width(printer_class),
        tape_px=tape_pixel_height(printer_class, tape_class),
    )


def default_local_cidr() -> str | None:
    """Best-effort local IPv4 network, capped at /24.

    Returns
    -------
    str or None
        CIDR such as ``"192.168.8.0/24"``, or None when no suitable
        interface was found.
    """
    try:
        import fcntl
        import struct
    except ImportError:  # pragma: no cover - non-Unix
        return None

    candidates: list[tuple[int, str]] = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for _, name in socket.if_nameindex():
            if name == "lo":
                continue
            try:
                packed = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", name.encode()[:15]))
                address = socket.inet_ntoa(packed[20:24])
                packed = fcntl.ioctl(sock.fileno(), 0x891B, struct.pack("256s", name.encode()[:15]))
                netmask = socket.inet_ntoa(packed[20:24])
            except OSError:
                continue
            network = ipaddress.ip_network(f"{address}/{netmask}", strict=False)
            if not network.is_private:
                continue
            if network.prefixlen < 24:
                network = ipaddress.ip_network(f"{address}/24", strict=False)
            # Prefer physical interfaces over bridges, VPNs and container networks.
            score = 0
            if name.startswith(("en", "eth", "wlan", "wlp", "wlan")):
                score = 2
            elif name.startswith(("br", "docker", "veth", "cni", "flannel", "tailscale")):
                score = -1
            candidates.append((score, str(network)))
    finally:
        sock.close()

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def scan_network(
    cidr: str | None = None,
    *,
    community: str = "public",
    version: str = "2c",
    timeout: float = 0.5,
    workers: int = 32,
) -> list[dict[str, Any]]:
    """Find Brother printers on a network with SNMP.

    Parameters
    ----------
    cidr : str or None, optional
        Network to scan. Defaults to the local /24 when omitted.
    community : str, default "public"
        SNMP read community.
    version : str, default "2c"
        SNMP version, ``"1"`` or ``"2c"``.
    timeout : float, default 0.5
        Timeout per host in seconds.
    workers : int, default 32
        Number of concurrent SNMP requests.

    Returns
    -------
    list of dict
        One entry per printer, with host, model, serial and media details.

    Raises
    ------
    ValueError
        If no CIDR is given and the local network cannot be determined, or
        the requested network is larger than /20.
    """
    network_text = cidr or default_local_cidr()
    if network_text is None:
        raise ValueError("cannot determine the local network; pass a CIDR to scan")
    network = ipaddress.ip_network(network_text, strict=False)
    if network.num_addresses > 4096:
        raise ValueError("refusing to scan more than /20; pass a smaller CIDR")

    def probe(host: ipaddress.IPv4Address | ipaddress.IPv6Address) -> dict[str, Any] | None:
        try:
            values = snmp_get(
                str(host),
                [
                    SNMP_HR_DEVICE_DESCR_OID,
                    SNMP_PRINTER_SERIAL_OID,
                    SNMP_VENDOR_STATUS_OID,
                ],
                community=community,
                version=version,
                timeout=timeout,
                retries=0,
            )
        except Exception:  # noqa: BLE001 - any failure means "not a printer"
            return None
        raw = values.get(SNMP_VENDOR_STATUS_OID)
        if not isinstance(raw, bytes) or len(raw) < 32:
            return None
        description = _decode_string(values.get(SNMP_HR_DEVICE_DESCR_OID))
        serial = _decode_string(values.get(SNMP_PRINTER_SERIAL_OID))
        status = enrich_status(parse_status_response(raw, model=description, serial=serial))
        return {
            "host": str(host),
            "model": status.model_name,
            "model_description": status.model,
            "serial": status.serial,
            "media_type": status.media_type_code,
            "media_type_label": status.media_type_label,
            "media_width_mm": status.media_width_mm,
            "ready": status.ready,
        }

    hosts = list(network.hosts())
    found: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(probe, host) for host in hosts]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                found.append(result)
    return sorted(found, key=lambda item: ipaddress.ip_address(item["host"]))


def discover_usb() -> list[dict[str, Any]]:
    """List Brother P-touch printers attached over USB.

    Returns
    -------
    list of dict
        One entry per device, with model, serial, product ID and whether the
        model is supported.

    Raises
    ------
    RuntimeError
        If pyusb is not installed or the USB devices cannot be enumerated.
    """
    try:
        import usb.core
    except ImportError as e:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "USB support requires the `pyusb` package. Install it via `pip install ptouch[usb]`."
        ) from e

    try:
        devices = usb.core.find(find_all=True, idVendor=0x04F9) or []
        found = []
        for device in devices:
            printer_class = printer_class_for_usb_product_id(device.idProduct)
            serial = None
            try:
                serial = device.serial_number
            except Exception:  # noqa: BLE001 - some devices deny the descriptor
                serial = None
            found.append(
                {
                    "model": printer_class.MODEL_NAME if printer_class else None,
                    "serial": serial,
                    "product_id": f"0x{device.idProduct:04x}",
                    "usable": printer_class is not None,
                }
            )
    except usb.core.USBError as e:
        raise RuntimeError(f"Cannot enumerate USB devices: {e}") from e
    return sorted(found, key=lambda item: (item["model"] or "", item["serial"] or ""))


def _decode_string(value: Any) -> str | None:
    """Decode an SNMP octet string value to text."""
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace").strip()
    if isinstance(value, str):
        return value.strip()
    return None
