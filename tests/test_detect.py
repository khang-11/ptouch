# SPDX-FileCopyrightText: 2024-2026 Nicolai Buchwitz <nb@tipi-net.de>
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Tests for printer and media detection helpers."""

from unittest.mock import MagicMock, patch

import pytest

from ptouch.detect import (
    PRINTER_TYPES,
    TAPE_WIDTHS,
    TUBE_WIDTHS,
    default_local_cidr,
    discover_usb,
    enrich_status,
    max_pixel_width,
    normalize_model_name,
    printer_class_for_model,
    printer_class_for_usb_product_id,
    resolve_media_class,
    scan_network,
    tape_pixel_height,
)
from ptouch.printer import MediaType
from ptouch.printers import PTP710BT, PTP900W, PTP950NW
from ptouch.status import (
    SNMP_HR_DEVICE_DESCR_OID,
    SNMP_PRINTER_SERIAL_OID,
    SNMP_VENDOR_STATUS_OID,
    PrinterStatus,
    parse_status_response,
)
from ptouch.tape import HeatShrinkTube3_1_21_0mm, Tape24mm

STATUS_READY = bytes.fromhex(
    "80 20 42 30 6F 30 04 00 00 00 18 01 00 00 00 0000 00 00 00 00 00 00 00 01 08 00 00 00 00 00 00"
)


class TestModelDetection:
    """Test model name and USB product ID lookup."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("Brother PT-P900W", "PT-P900W"),
            ("PT-P900W", "PT-P900W"),
            ("P900W", "P900W"),
            ("Brother PT-P710BT", "PT-P710BT"),
            ("", None),
            (None, None),
        ],
    )
    def test_normalize_model_name(self, model: str | None, expected: str | None) -> None:
        """Model descriptions are normalized to the short form."""
        assert normalize_model_name(model) == expected

    def test_printer_class_for_model(self) -> None:
        """Model names resolve to the matching printer class."""
        assert printer_class_for_model("Brother PT-P900W") is PTP900W
        assert printer_class_for_model("P900W") is PTP900W
        assert printer_class_for_model("PT-P710BT") is PTP710BT
        assert printer_class_for_model("nonsense") is None

    def test_printer_class_for_usb_product_id(self) -> None:
        """USB product IDs resolve to the matching printer class."""
        assert printer_class_for_usb_product_id(0x2085) is PTP900W
        assert printer_class_for_usb_product_id(0x2086) is PTP950NW
        assert printer_class_for_usb_product_id(0xFFFF) is None

    def test_printer_types_expose_only_supported_models(self) -> None:
        """The short-name map contains the supported printers."""
        assert PRINTER_TYPES["P900W"] is PTP900W
        assert PRINTER_TYPES["P710BT"] is PTP710BT


class TestMediaDetection:
    """Test media width/type lookup and status enrichment."""

    def test_resolve_laminated_tape(self) -> None:
        """Laminated media resolves to a TZe tape class."""
        assert resolve_media_class(24, MediaType.LAMINATED_TAPE.value) is Tape24mm
        assert resolve_media_class(3.5, 1) is TAPE_WIDTHS[3.5]

    def test_resolve_heat_shrink_tube(self) -> None:
        """Heat shrink media resolves to a tube class."""
        assert (
            resolve_media_class(21, MediaType.HEATSHRINK_TUBE_31.value) is HeatShrinkTube3_1_21_0mm
        )
        assert resolve_media_class(5.8, MediaType.HEATSHRINK_TUBE_21.value) is TUBE_WIDTHS[5.8]

    def test_resolve_unknown_media(self) -> None:
        """Unknown widths or no media return None."""
        assert resolve_media_class(None) is None
        assert resolve_media_class(0) is None
        assert resolve_media_class(99) is None

    def test_pixel_widths(self) -> None:
        """Printer and tape pixel sizes come from the pin configuration."""
        assert max_pixel_width(PTP900W) == 454
        assert tape_pixel_height(PTP900W, Tape24mm) == 320
        assert tape_pixel_height(PTP900W, None) is None

    def test_enrich_status(self) -> None:
        """Status is enriched with model-specific sizes."""
        status = parse_status_response(STATUS_READY, model="Brother PT-P900W", serial="123")
        enriched = enrich_status(status)
        assert enriched.dpi == 360
        assert enriched.max_px == 454
        assert enriched.tape_px == 320
        assert enriched.serial == "123"
        assert enriched.ready is True

    def test_enrich_status_with_explicit_class(self) -> None:
        """An explicit printer class is used when the model is unknown."""
        status = parse_status_response(STATUS_READY)
        enriched = enrich_status(status, PTP900W)
        assert enriched.model == "PT-P900W"
        assert enriched.dpi == 360


class TestDiscovery:
    """Test network scanning and USB enumeration."""

    def test_default_local_cidr(self) -> None:
        """The local CIDR is either discovered or None on unsupported systems."""
        cidr = default_local_cidr()
        assert cidr is None or cidr.endswith(("/24", "/25", "/26", "/27", "/28", "/29", "/30"))

    def test_scan_network(self) -> None:
        """Hosts that answer with a Brother status are returned."""
        hosts: list[str] = []

        def fake_snmp_get(host: str, oids: list[str], **kwargs: object) -> dict[str, object]:
            del oids, kwargs
            hosts.append(host)
            if host.endswith(".1"):
                return {
                    SNMP_HR_DEVICE_DESCR_OID: b"Brother PT-P900W",
                    SNMP_PRINTER_SERIAL_OID: b"M2G127057",
                    SNMP_VENDOR_STATUS_OID: STATUS_READY,
                }
            raise OSError("timeout")

        with patch("ptouch.detect.snmp_get", side_effect=fake_snmp_get):
            printers = scan_network("192.0.2.0/30", workers=2)

        assert hosts == ["192.0.2.1", "192.0.2.2"]
        assert len(printers) == 1
        assert printers[0]["host"] == "192.0.2.1"
        assert printers[0]["model"] == "PT-P900W"
        assert printers[0]["serial"] == "M2G127057"
        assert printers[0]["ready"] is True

    def test_scan_network_rejects_large_networks(self) -> None:
        """A scan larger than /20 is refused."""
        with pytest.raises(ValueError, match="/20"):
            scan_network("10.0.0.0/8")

    def test_scan_network_without_local_network(self) -> None:
        """A missing local network needs an explicit CIDR."""
        with patch("ptouch.detect.default_local_cidr", return_value=None):
            with pytest.raises(ValueError, match="cannot determine"):
                scan_network()

    def test_discover_usb(self) -> None:
        """USB devices are mapped to models and serials."""
        device = MagicMock()
        device.idProduct = 0x2085
        device.serial_number = "A1B2"
        with patch("usb.core.find", return_value=[device]):
            printers = discover_usb()
        assert printers == [
            {"model": "PT-P900W", "serial": "A1B2", "product_id": "0x2085", "usable": True}
        ]

    def test_discover_usb_unsupported_model(self) -> None:
        """Unknown product IDs are listed as not usable."""
        device = MagicMock()
        device.idProduct = 0xFFFF
        device.serial_number = "X"
        with patch("usb.core.find", return_value=[device]):
            printers = discover_usb()
        assert printers[0]["model"] is None
        assert printers[0]["usable"] is False

    def test_discover_usb_without_pyusb(self) -> None:
        """Missing pyusb produces a clear error."""
        with patch.dict("sys.modules", {"usb": None, "usb.core": None}):
            with pytest.raises(RuntimeError, match="pyusb"):
                discover_usb()

    def test_discover_usb_error(self) -> None:
        """USB enumeration errors are reported."""
        import usb.core

        with patch("usb.core.find", side_effect=usb.core.USBError("denied")):
            with pytest.raises(RuntimeError, match="Cannot enumerate"):
                discover_usb()


class TestStatusHelpers:
    """Test the extra status properties."""

    def test_model_name(self) -> None:
        """The short model name is extracted from the description."""
        assert PrinterStatus(model="Brother PT-P900W").model_name == "PT-P900W"
        assert PrinterStatus(model="P900W").model_name == "P900W"
        assert PrinterStatus().model_name is None

    def test_error_code(self) -> None:
        """Error information bytes form the 16-bit error code."""
        status = parse_status_response(STATUS_READY)
        assert status.error_code == 0
        bad = bytearray(STATUS_READY)
        bad[8] = 0x04
        bad[9] = 0x10
        assert parse_status_response(bytes(bad)).error_code == 0x1004

    def test_ready(self) -> None:
        """Ready requires a connection, media and no error."""
        status = parse_status_response(STATUS_READY)
        assert status.ready is True
        assert PrinterStatus(connected=False, media_width_mm=24).ready is False
        assert PrinterStatus(media_type_code=0, media_width_mm=0).ready is False
        assert PrinterStatus(media_width_mm=24, errors=("cutter jam",)).ready is False

    def test_connected_defaults(self) -> None:
        """Parsed status is connected; an empty status can be disconnected."""
        assert parse_status_response(STATUS_READY).connected is True
        assert PrinterStatus(connected=False).device_status == "NOT CONNECTED"
