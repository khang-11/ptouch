# SPDX-FileCopyrightText: 2024-2026 Nicolai Buchwitz <nb@tipi-net.de>
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Tests for the ptouch.status module."""

import socket
import sys
from unittest.mock import MagicMock, patch

import pytest

from ptouch.__main__ import main
from ptouch.connection import (
    Connection,
    ConnectionNetwork,
    ConnectionUSB,
    PrinterConnectionError,
)
from ptouch.printer import MediaType
from ptouch.printers import PTP900W
from ptouch.status import (
    SNMP_VENDOR_STATUS_OID,
    PrinterStatus,
    SnmpError,
    StatusType,
    _build_get_request,
    _decode_oid,
    _encode_integer,
    _encode_oid,
    _parse_snmp_response,
    _read_tlv,
    _tlv,
    _udp_exchange,
    parse_media_name,
    parse_status_response,
    query_status_snmp,
    snmp_get,
)

# Status block captured from a PT-P900W with 24 mm laminated tape loaded:
# battery 4, no errors, media width 0x18, media type 0x01 (laminated).
STATUS_READY = bytes.fromhex(
    "80 20 42 30 6F 30 04 00 00 00 18 01 00 00 00 0000 00 00 00 00 00 00 00 01 08 00 00 00 00 00 00"
)


def _with_byte(data: bytes, offset: int, value: int) -> bytes:
    """Return a copy of *data* with one byte replaced."""
    result = bytearray(data)
    result[offset] = value
    return bytes(result)


def _encode_snmp_response(
    request_id: int,
    varbinds: list[tuple[str, int, bytes]],
    *,
    error_status: int = 0,
) -> bytes:
    """Build a minimal SNMP GetResponse for tests."""
    encoded_varbinds = b"".join(
        _tlv(0x30, _encode_oid(oid) + _tlv(tag, value)) for oid, tag, value in varbinds
    )
    pdu = _tlv(
        0xA2,
        _encode_integer(request_id)
        + _encode_integer(error_status)
        + _encode_integer(0)
        + _tlv(0x30, encoded_varbinds),
    )
    return _tlv(
        0x30,
        _encode_integer(1) + _tlv(0x04, b"public") + pdu,
    )


def _request_id(payload: bytes) -> int:
    """Extract the request id from an encoded SNMP GET request."""
    _, message, _ = _read_tlv(payload, 0)
    pos = 0
    _, _, pos = _read_tlv(message, pos)  # version
    _, _, pos = _read_tlv(message, pos)  # community
    _, pdu, pos = _read_tlv(message, pos)
    _, raw_id, _ = _read_tlv(pdu, 0)
    return int.from_bytes(raw_id, "big", signed=True)


class StatusConnection(Connection):
    """Connection returning a canned status response."""

    def __init__(self, response: bytes) -> None:
        self.response = response
        self.written = b""

    def connect(self, printer: object) -> None:
        """Pretend to connect."""
        del printer

    def write(self, payload: bytes) -> None:
        """Capture written data."""
        self.written += payload

    def read(self, num_bytes: int = 1024) -> bytes:
        """Return the next chunk of the canned response."""
        chunk = self.response[:num_bytes]
        self.response = self.response[num_bytes:]
        return chunk

    def close(self) -> None:
        """No-op close."""


class TestParseStatusResponse:
    """Test parsing of the 32-byte status block."""

    def test_ready_status(self) -> None:
        """A ready printer with 24 mm laminated tape parses correctly."""
        status = parse_status_response(STATUS_READY, model="Brother PT-P900W", serial="1234")

        assert status.model == "Brother PT-P900W"
        assert status.serial == "1234"
        assert status.media_width_mm == 24.0
        assert status.media_type is MediaType.LAMINATED_TAPE
        assert status.media_type_label == "laminated tape"
        assert status.media_length_mm == 0
        assert status.battery_level == 4
        assert status.errors == ()
        assert status.status_type == StatusType.REPLY_TO_STATUS_REQUEST
        assert status.model_code == 0x6F
        assert status.device_status == "READY"
        assert status.media_status == "Not Empty"
        assert status.is_error is False
        assert status.source == "vendor-status"
        assert status.raw == STATUS_READY

    def test_heat_shrink_tube(self) -> None:
        """Heat shrink media type is decoded."""
        status = parse_status_response(_with_byte(STATUS_READY, 11, 0x17))
        assert status.media_type is MediaType.HEATSHRINK_TUBE_31
        assert status.media_type_label == "heat-shrink tube (3:1)"

    def test_unknown_media_type(self) -> None:
        """Unknown media type codes are reported with their raw value."""
        status = parse_status_response(_with_byte(STATUS_READY, 11, 0x42))
        assert status.media_type is None
        assert status.media_type_label == "unknown (0x42)"

    def test_no_media(self) -> None:
        """A no-media status reports an empty media status."""
        data = _with_byte(_with_byte(STATUS_READY, 10, 0), 11, 0)
        status = parse_status_response(data)
        assert status.media_width_mm is None
        assert status.media_type is MediaType.NO_MEDIA
        assert status.media_status == "Empty"

    def test_error_flags(self) -> None:
        """Error flag bytes are decoded into messages."""
        data = _with_byte(_with_byte(STATUS_READY, 8, 0x04), 9, 0x11)
        status = parse_status_response(data)
        assert status.errors == ("cutter jam", "wrong media", "cover open")
        assert status.device_status == "ERROR"
        assert status.is_error is True

    def test_error_status_type(self) -> None:
        """Status type 2 marks the printer as failed even without flags."""
        status = parse_status_response(_with_byte(STATUS_READY, 18, 0x02))
        assert status.status_type == StatusType.ERROR_OCCURRED
        assert status.device_status == "ERROR"
        assert status.is_error is True

    def test_too_short(self) -> None:
        """Status blocks shorter than 32 bytes are rejected."""
        with pytest.raises(ValueError, match="at least 32 bytes"):
            parse_status_response(STATUS_READY[:31])

    def test_invalid_header(self) -> None:
        """Status blocks without the Brother header are rejected."""
        with pytest.raises(ValueError, match="Invalid status header"):
            parse_status_response(_with_byte(STATUS_READY, 0, 0x00))

    def test_extra_bytes_ignored(self) -> None:
        """Trailing bytes after the 32-byte block are ignored."""
        status = parse_status_response(STATUS_READY + b"\xff\xff")
        assert status.raw == STATUS_READY


class TestPrinterStatusProperties:
    """Test derived PrinterStatus properties."""

    def test_empty_status(self) -> None:
        """An empty PrinterStatus reports unknown values."""
        status = PrinterStatus()
        assert status.media_type is None
        assert status.media_type_label == "unknown"
        assert status.device_status == "UNKNOWN"
        assert status.media_status == "Unknown"
        assert status.is_error is False

    def test_display_status_is_trimmed(self) -> None:
        """Display text from the Printer MIB is stripped."""
        status = PrinterStatus(display_status="READY           ")
        assert status.device_status == "READY"

    def test_display_status_error(self) -> None:
        """An ERROR display text marks the status as an error."""
        status = PrinterStatus(display_status="ERROR           ")
        assert status.device_status == "ERROR"
        assert status.is_error is True


class TestParseMediaName:
    """Test parsing of Printer MIB media names."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ('24mm(0.94")', 24.0),
            ('3.5mm(0.14")', 3.5),
            ("36 mm", 36.0),
            ("no media", None),
            ("", None),
            (None, None),
        ],
    )
    def test_parse_media_name(self, name: str | None, expected: float | None) -> None:
        """Media widths are extracted from the media name string."""
        assert parse_media_name(name) == expected


class TestSnmpEncoding:
    """Test the minimal SNMP BER encoder and decoder."""

    def test_encode_oid(self) -> None:
        """OIDs are encoded with the expected BER bytes."""
        assert _encode_oid("1.3.6.1.4.1.2435") == bytes.fromhex("06 07 2b 06 01 04 01 93 03")

    @pytest.mark.parametrize(
        "oid",
        [
            "1.3.6.1.4.1.2435.3.3.9.1.6.1.0",
            "1.3.6.1.2.1.43.8.2.1.12.1.1",
            "1.3.6.1.2.1.25.3.2.1.3.1",
        ],
    )
    def test_oid_round_trip(self, oid: str) -> None:
        """Encoding and decoding an OID returns the original OID."""
        _, payload, _ = _read_tlv(_encode_oid(oid), 0)
        assert _decode_oid(payload) == oid

    def test_invalid_oid(self) -> None:
        """An OID with too few arcs is rejected."""
        with pytest.raises(ValueError, match="at least two arcs"):
            _encode_oid("1")

    def test_build_get_request(self) -> None:
        """A GET request can be decoded back into its request id."""
        payload, request_id = _build_get_request(
            [SNMP_VENDOR_STATUS_OID], community="public", version=1
        )
        assert _request_id(payload) == request_id

    def test_parse_response(self) -> None:
        """A GetResponse with one varbind is parsed correctly."""
        request_id = 42
        response = _encode_snmp_response(request_id, [(SNMP_VENDOR_STATUS_OID, 0x04, STATUS_READY)])
        error_status, error_index, varbinds = _parse_snmp_response(response, request_id)

        assert error_status == 0
        assert error_index == 0
        assert varbinds == [(SNMP_VENDOR_STATUS_OID, 0x04, STATUS_READY)]

    def test_parse_response_rejects_wrong_request_id(self) -> None:
        """Responses for another request id are rejected."""
        response = _encode_snmp_response(1, [(SNMP_VENDOR_STATUS_OID, 0x04, STATUS_READY)])
        with pytest.raises(SnmpError, match="request-id"):
            _parse_snmp_response(response, 2)

    def test_parse_response_rejects_short_message(self) -> None:
        """A truncated message is rejected."""
        with pytest.raises(SnmpError):
            _parse_snmp_response(b"\x30\x00", 1)


class TestSnmpGet:
    """Test the high-level SNMP GET helper."""

    def test_snmp_get_returns_values(self) -> None:
        """Values from the response are mapped back to the requested OIDs."""

        def fake_exchange(
            payload: bytes, host: str, port: int, timeout: float, retries: int
        ) -> bytes:
            del host, port, timeout, retries
            request_id = _request_id(payload)
            return _encode_snmp_response(
                request_id,
                [
                    (SNMP_VENDOR_STATUS_OID, 0x04, STATUS_READY),
                    ("1.3.6.1.2.1.43.5.1.1.17.1", 0x04, b"M2G127057"),
                ],
            )

        with patch("ptouch.status._udp_exchange", side_effect=fake_exchange):
            result = snmp_get(
                "192.0.2.1",
                [SNMP_VENDOR_STATUS_OID, "1.3.6.1.2.1.43.5.1.1.17.1"],
            )

        assert result[SNMP_VENDOR_STATUS_OID] == STATUS_READY
        assert result["1.3.6.1.2.1.43.5.1.1.17.1"] == b"M2G127057"

    def test_snmp_get_missing_oid(self) -> None:
        """Missing varbinds are returned as None."""

        def fake_exchange(
            payload: bytes, host: str, port: int, timeout: float, retries: int
        ) -> bytes:
            del host, port, timeout, retries
            return _encode_snmp_response(_request_id(payload), [])

        with patch("ptouch.status._udp_exchange", side_effect=fake_exchange):
            result = snmp_get("192.0.2.1", [SNMP_VENDOR_STATUS_OID])
        assert result == {SNMP_VENDOR_STATUS_OID: None}

    def test_snmp_v1_no_such_name(self) -> None:
        """SNMPv1 noSuchName responses are treated as missing OIDs."""

        def fake_exchange(
            payload: bytes, host: str, port: int, timeout: float, retries: int
        ) -> bytes:
            del host, port, timeout, retries
            return _encode_snmp_response(_request_id(payload), [], error_status=2)

        with patch("ptouch.status._udp_exchange", side_effect=fake_exchange):
            result = snmp_get("192.0.2.1", [SNMP_VENDOR_STATUS_OID], version="1")
        assert result == {SNMP_VENDOR_STATUS_OID: None}

    def test_snmp_get_error_status(self) -> None:
        """Non-zero SNMP error statuses other than noSuchName are raised."""

        def fake_exchange(
            payload: bytes, host: str, port: int, timeout: float, retries: int
        ) -> bytes:
            del host, port, timeout, retries
            return _encode_snmp_response(_request_id(payload), [], error_status=5)

        with patch("ptouch.status._udp_exchange", side_effect=fake_exchange):
            with pytest.raises(SnmpError, match="status 5"):
                snmp_get("192.0.2.1", [SNMP_VENDOR_STATUS_OID])

    def test_udp_exchange_timeout(self) -> None:
        """A socket timeout is reported as an SNMP error."""
        mock_socket = MagicMock()
        mock_socket.recvfrom.side_effect = socket.timeout("timed out")

        with patch("ptouch.status.socket.socket", return_value=mock_socket):
            with pytest.raises(SnmpError, match="No SNMP response"):
                _udp_exchange(b"payload", "192.0.2.1", 161, 0.1, retries=1)

    def test_invalid_version(self) -> None:
        """Unsupported SNMP versions are rejected."""
        with pytest.raises(ValueError, match="Unsupported SNMP version"):
            snmp_get("192.0.2.1", [SNMP_VENDOR_STATUS_OID], version="3")


class TestQueryStatusSnmp:
    """Test the network status query."""

    def test_vendor_status_is_used(self) -> None:
        """The vendor 32-byte status block is preferred when available."""
        varbinds = {
            SNMP_VENDOR_STATUS_OID: STATUS_READY,
            "1.3.6.1.2.1.25.3.2.1.3.1": b"Brother PT-P900W",
            "1.3.6.1.2.1.43.5.1.1.17.1": b"M2G127057",
            "1.3.6.1.2.1.43.8.2.1.12.1.1": b'24mm(0.94")',
            "1.3.6.1.2.1.43.16.5.1.2.1.1": b"READY           ",
        }
        with patch("ptouch.status.snmp_get", return_value=varbinds):
            status = query_status_snmp("192.0.2.1", community="secret", version="1")

        assert status.source == "vendor-status"
        assert status.model == "Brother PT-P900W"
        assert status.serial == "M2G127057"
        assert status.media_width_mm == 24.0

    def test_printer_mib_fallback(self) -> None:
        """The standard Printer MIB is used when the vendor OID is absent."""
        varbinds = {
            SNMP_VENDOR_STATUS_OID: None,
            "1.3.6.1.2.1.25.3.2.1.3.1": b"Brother PT-P900W",
            "1.3.6.1.2.1.43.5.1.1.17.1": b"M2G127057",
            "1.3.6.1.2.1.43.8.2.1.12.1.1": b'12mm(0.47")',
            "1.3.6.1.2.1.43.16.5.1.2.1.1": b"READY           ",
        }
        with patch("ptouch.status.snmp_get", return_value=varbinds):
            status = query_status_snmp("192.0.2.1")

        assert status.source == "printer-mib"
        assert status.media_width_mm == 12.0
        assert status.media_type is None
        assert status.device_status == "READY"

    def test_no_status_information(self) -> None:
        """An SNMP error is raised when the printer exposes no status data."""
        varbinds = dict.fromkeys(
            [
                SNMP_VENDOR_STATUS_OID,
                "1.3.6.1.2.1.25.3.2.1.3.1",
                "1.3.6.1.2.1.43.5.1.1.17.1",
                "1.3.6.1.2.1.43.8.2.1.12.1.1",
                "1.3.6.1.2.1.43.16.5.1.2.1.1",
            ]
        )
        with patch("ptouch.status.snmp_get", return_value=varbinds):
            with pytest.raises(SnmpError, match="does not expose status"):
                query_status_snmp("192.0.2.1")


class TestPrinterGetStatus:
    """Test LabelPrinter.get_status over USB and network connections."""

    def test_usb_status(self) -> None:
        """USB status sends ESC i S and parses the response."""
        connection = StatusConnection(STATUS_READY)
        printer = PTP900W(connection)

        status = printer.get_status()

        assert connection.written.endswith(b"\x1b\x69\x53")
        assert status.media_width_mm == 24.0

    def test_network_status_uses_snmp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Network status is queried over SNMP instead of the raw socket."""
        connection = ConnectionNetwork("192.0.2.1")
        monkeypatch.setattr(connection, "connect", lambda printer: None)
        monkeypatch.setattr(connection, "write", lambda payload: None)
        printer = PTP900W(connection)
        expected = parse_status_response(STATUS_READY)

        with patch("ptouch.status.query_status_snmp", return_value=expected) as mock_query:
            status = printer.get_status(community="secret", snmp_version="1")

        assert status is expected
        mock_query.assert_called_once_with(
            "192.0.2.1",
            community="secret",
            version="1",
            timeout=3.0,
        )

    def test_snmp_error_is_a_connection_error(self) -> None:
        """SnmpError can be caught as a printer connection error."""
        assert issubclass(SnmpError, PrinterConnectionError)


class TestUsbStatusRead:
    """Test the USB endpoint read used for status queries."""

    def test_usb_read(self) -> None:
        """ConnectionUSB.read reads from the IN endpoint."""
        connection = ConnectionUSB(timeout=2.5)
        connection._ep_in = MagicMock()
        connection._ep_in.read.return_value = bytearray(STATUS_READY)

        assert connection.read(32) == STATUS_READY
        connection._ep_in.read.assert_called_once_with(32, timeout=2500)

    def test_usb_read_not_connected(self) -> None:
        """Reading without a connected endpoint raises a connection error."""
        connection = ConnectionUSB()
        with pytest.raises(PrinterConnectionError, match="Not connected"):
            connection.read(32)


class TestStatusCli:
    """Test the --status command line handling."""

    def test_status_network(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A network status query prints the parsed status."""
        monkeypatch.setattr(sys, "argv", ["ptouch", "--status", "--host", "192.0.2.1"])
        expected = parse_status_response(STATUS_READY, model="Brother PT-P900W")

        with patch("ptouch.__main__.query_status_snmp", return_value=expected) as mock_query:
            assert main() == 0

        mock_query.assert_called_once_with("192.0.2.1", community="public", version="2c")
        output = capsys.readouterr().out
        assert "Device status:  READY" in output
        assert "Media type:     laminated tape" in output
        assert 'Media width:    24 mm (0.94")' in output

    def test_status_json(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--json prints a JSON object."""
        monkeypatch.setattr(sys, "argv", ["ptouch", "--status", "--host", "192.0.2.1", "--json"])
        with patch(
            "ptouch.__main__.query_status_snmp", return_value=parse_status_response(STATUS_READY)
        ):
            assert main() == 0

        output = capsys.readouterr().out
        assert '"device_status": "READY"' in output
        assert '"media_width_mm": 24.0' in output

    def test_usb_status_requires_printer(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """USB status queries need a printer model for the product id."""
        monkeypatch.setattr(sys, "argv", ["ptouch", "--status", "--usb"])
        assert main() == 1
        assert "--printer is required" in capsys.readouterr().err

    def test_status_rejects_text(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--status cannot be combined with label content."""
        monkeypatch.setattr(sys, "argv", ["ptouch", "--status", "--host", "192.0.2.1", "Hello"])
        assert main() == 1
        assert "Cannot combine --status" in capsys.readouterr().err

    def test_json_requires_status(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--json is only valid together with --status."""
        monkeypatch.setattr(sys, "argv", ["ptouch", "--json", "--host", "192.0.2.1"])
        assert main() == 1
        assert "--json requires --status" in capsys.readouterr().err

    def test_status_rejects_media_options(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--status cannot be combined with tape or tube options."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["ptouch", "--status", "--host", "192.0.2.1", "--tape-width", "12"],
        )
        assert main() == 1
        assert "Cannot combine --status" in capsys.readouterr().err
