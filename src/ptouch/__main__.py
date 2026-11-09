# SPDX-FileCopyrightText: 2024-2026 Nicolai Buchwitz <nb@tipi-net.de>
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""CLI interface for Brother P-touch label printers."""

import argparse
import json
import sys
from typing import Sequence

from PIL import Image, ImageFont

from . import (
    Align,
    ConnectionNetwork,
    ConnectionUSB,
    Label,
    PrinterStatus,
    TextLabel,
    __version__,
    parse_usb_uri,
    query_status_snmp,
)
from .detect import (
    PRINTER_TYPES,
    TAPE_WIDTHS,
    TUBE_WIDTHS,
    discover_usb,
    printer_class_for_model,
    printer_class_for_usb_product_id,
    resolve_media_class,
    scan_network,
)
from .printer import LabelPrinter
from .tape import Tape

# Mapping of alignment strings to Align flags
ALIGN_HORIZONTAL = {
    "left": Align.LEFT,
    "center": Align.HCENTER,
    "right": Align.RIGHT,
}

ALIGN_VERTICAL = {
    "top": Align.TOP,
    "center": Align.VCENTER,
    "bottom": Align.BOTTOM,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="ptouch",
        description="Print labels on Brother P-touch printers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Print text label via network; printer model and tape are detected
  ptouch "Hello World" --host 192.168.1.100

  # Print with an explicit model and tape (skips detection)
  ptouch "Hello World" --host 192.168.1.100 --printer P900 --tape-width 36

  # Print with custom font
  ptouch "Hello World" --host 192.168.1.100 --font /path/to/font.ttf

  # Print multiple labels (half-cut between, full cut after last)
  ptouch "Label 1" "Label 2" "Label 3" --host 192.168.1.100 --full-cut

  # Print image label via USB, auto-detecting the printer
  ptouch --image logo.png --usb

  # Print an image exactly as supplied, without a feed margin
  ptouch --image logo.png --host 192.168.1.100 --no-margin

  # Print via USB with a specific device (vendor:product/serial)
  ptouch "Test" --usb usb://:0x2086/A1B2C3D4E5

  # Print on heat shrink tube (2:1 series, 5.8mm)
  ptouch "Cable 1" --host 192.168.1.100 --tube-width 5.8

  # Query printer status (media type/width and errors) via network
  ptouch --status --host 192.168.1.100

  # Query printer status via USB (printer detected automatically)
  ptouch --status --usb

  # Query status as JSON for scripting
  ptouch --status --host 192.168.1.100 --json

  # Find Brother printers on the local network
  ptouch --scan 192.168.0/24

  # List printers attached over USB
  ptouch --list-connected
""",
    )

    # Content (text or image)
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to print. Multiple strings create multiple labels with half-cut between.",
    )
    parser.add_argument(
        "--image",
        "-i",
        metavar="FILE",
        help="Image file to print instead of text",
    )

    # Connection
    conn_group = parser.add_mutually_exclusive_group()
    conn_group.add_argument(
        "--host",
        "-H",
        metavar="IP",
        help="Printer IP address or hostname for network connection",
    )
    conn_group.add_argument(
        "--usb",
        nargs="?",
        const=True,
        default=None,
        metavar="URI",
        help="Use USB connection. Optional URI: usb://[vendor:]product[/serial] "
        "(e.g., usb://:0x2086/A1B2C3D4E5)",
    )
    parser.add_argument(
        "--serial",
        metavar="SERIAL",
        help="USB serial number of the printer to use when several are attached",
    )

    # Printer and tape/tube
    parser.add_argument(
        "--printer",
        "-p",
        choices=list(PRINTER_TYPES.keys()),
        help="Printer model (auto-detected when omitted)",
    )

    media_group = parser.add_mutually_exclusive_group()
    media_group.add_argument(
        "--tape-width",
        "-t",
        type=float,
        choices=list(TAPE_WIDTHS.keys()),
        help="Laminated tape width in mm (auto-detected when omitted)",
    )
    media_group.add_argument(
        "--tube-width",
        "-T",
        type=float,
        choices=list(TUBE_WIDTHS.keys()),
        help="Heat shrink tube diameter in mm (2:1: 5.8/8.8/11.7/17.7/23.6, "
        "3:1: 5.2/9.0/11.2/21.0/31.0)",
    )

    # Font options
    parser.add_argument(
        "--font",
        "-f",
        metavar="PATH",
        help="Path to TrueType font file (uses PIL default font if not specified)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        metavar="PX",
        help="Font size in pixels (disables auto-sizing to 80%% of print height)",
    )

    # Alignment
    parser.add_argument(
        "--align",
        "-a",
        nargs=2,
        metavar=("H", "V"),
        default=["center", "center"],
        help="Alignment: left|center|right top|center|bottom (default: center center)",
    )

    # Print options
    parser.add_argument(
        "--high-resolution",
        action="store_true",
        help="Enable high resolution mode",
    )
    parser.add_argument(
        "--margin",
        "-m",
        type=float,
        metavar="MM",
        help="Feed margin in mm (default: 2mm); 0 omits the margin command",
    )
    parser.add_argument(
        "--no-margin",
        action="store_true",
        help="Do not send a feed margin command; print images exactly as supplied",
    )
    parser.add_argument(
        "--vertical-offset",
        type=int,
        metavar="PX",
        help="Pins to skip before the printable area (default: centre it on the tape)",
    )
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable TIFF compression",
    )
    parser.add_argument(
        "--full-cut",
        action="store_true",
        help="Use full cuts between labels instead of half-cuts (default: half-cut)",
    )
    parser.add_argument(
        "--chain",
        action="store_true",
        help="Chain printing: do not feed and cut after the final label",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the printed image (for clear tape)",
    )
    parser.add_argument(
        "--precut",
        action="store_true",
        help=(
            "Eject the ~24 mm leader between the print head and the cutter "
            "as a small scrap before the first label, so the real label "
            "starts at a fresh cut edge with no blank leading tape."
        ),
    )
    parser.add_argument(
        "--copies",
        "-c",
        type=int,
        default=1,
        metavar="N",
        help="Number of copies to print (default: 1)",
    )
    parser.add_argument(
        "--width",
        "-w",
        type=float,
        metavar="MM",
        help="Fixed label width in mm (default: auto-sized to content)",
    )

    # Status, discovery and JSON output
    parser.add_argument(
        "--status",
        action="store_true",
        help="Query printer status (media type/width, errors) instead of printing",
    )
    parser.add_argument(
        "--list-connected",
        action="store_true",
        help="List Brother printers attached over USB",
    )
    parser.add_argument(
        "--scan",
        nargs="?",
        const=True,
        default=None,
        metavar="CIDR",
        help="Find Brother printers on the network with SNMP (default: local /24)",
    )
    parser.add_argument(
        "--scan-timeout",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="Timeout per host when scanning (default: 0.5)",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=32,
        metavar="N",
        help="Concurrent SNMP requests when scanning (default: 32)",
    )
    parser.add_argument(
        "--snmp-community",
        metavar="COMMUNITY",
        default="public",
        help="SNMP read community for network status and scans (default: public)",
    )
    parser.add_argument(
        "--snmp-version",
        choices=["1", "2c"],
        default="2c",
        help="SNMP version for network status and scans (default: 2c)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print status, scan or USB list as JSON",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser.parse_args()


def create_text_labels(
    texts: Sequence[str],
    tape_class: type,
    font: str | ImageFont.FreeTypeFont,
    align: Align,
    font_size: int | None = None,
    min_width_mm: float | None = None,
    fixed_width_mm: float | None = None,
    auto_size: bool = True,
) -> list[TextLabel]:
    """Create TextLabel instances for multiple text strings.

    Parameters
    ----------
    texts : Sequence[str]
        List of text strings to create labels for.
    tape_class : type
        Tape class to use for all labels.
    font : str or ImageFont.FreeTypeFont
        Path to TrueType font file or ImageFont object.
    align : Align
        Text alignment flags.
    font_size : int or None
        Font size in pixels (only used when auto_size=False).
    min_width_mm : float or None
        Minimum label width in millimeters.
    fixed_width_mm : float or None
        Fixed label width in millimeters (locks the label to this width).
    auto_size : bool
        If True, auto-size font to 80% of print height.

    Returns
    -------
    list[TextLabel]
        List of TextLabel instances.
    """
    return [
        TextLabel(
            text,
            tape_class,
            font=font,
            font_size=font_size,
            align=align,
            min_width_mm=min_width_mm,
            fixed_width_mm=fixed_width_mm,
            auto_size=auto_size,
        )
        for text in texts
    ]


def _format_media_width(width_mm: float | None) -> str:
    """Format a media width in millimetres and inches."""
    if width_mm is None:
        return "unknown"
    return f'{width_mm:g} mm ({width_mm / 25.4:.2f}")'


def _status_message(status: PrinterStatus) -> str:
    """Human-readable explanation when the printer is not ready."""
    if status.message:
        return status.message
    if not status.connected:
        return "No printer found."
    if status.errors:
        return "; ".join(status.errors)
    if status.media_type is not None and status.media_width_mm is None:
        return "No tape cassette inserted."
    if status.media_width_mm is None or status.media_width_mm <= 0:
        return "No tape cassette inserted."
    if not status.ready:
        return f"The printer is not ready ({status.device_status})."
    return ""


def _status_to_dict(status: PrinterStatus) -> dict[str, object]:
    """Convert a PrinterStatus to the JSON-friendly shape used by the CLI."""
    return {
        "connected": status.connected,
        "ready": status.ready,
        "model": status.model_name,
        "model_description": status.model,
        "model_code": status.model_code,
        "serial": status.serial,
        "dpi": status.dpi,
        "max_px": status.max_px,
        "tape_px": status.tape_px,
        "media_type": status.media_type_code,
        "media_type_label": status.media_type_label,
        "media_width_mm": status.media_width_mm,
        "media_length_mm": status.media_length_mm,
        "tape_color": status.tape_color,
        "text_color": status.text_color,
        "error_code": status.error_code,
        "errors": list(status.errors),
        "battery_level": status.battery_level,
        "device_status": status.device_status,
        "media_status": status.media_status,
        "message": _status_message(status),
        "source": status.source,
        "ptouch_version": __version__,
    }


def print_status(status: PrinterStatus, *, as_json: bool = False) -> None:
    """Print printer status as human-readable text or JSON."""
    if as_json:
        print(json.dumps(_status_to_dict(status), indent=2))
        return

    lines = [
        f"Printer:        {status.model or 'unknown'}",
        f"Serial:         {status.serial or 'unknown'}",
        f"Device status:  {status.device_status}",
        f"Media status:   {status.media_status}",
        f"Media type:     {status.media_type_label}",
        f"Media width:    {_format_media_width(status.media_width_mm)}",
    ]
    if status.dpi is not None:
        lines.append(f"Resolution:     {status.dpi} dpi")
    if status.tape_px is not None:
        lines.append(f"Tape height:    {status.tape_px} px")
    if status.media_length_mm:
        lines.append(f"Media length:   {status.media_length_mm} mm")
    if status.battery_level is not None:
        lines.append(f"Battery:        {status.battery_level}")
    message = _status_message(status)
    lines.append(f"Message:        {message or 'none'}")
    print("\n".join(lines))


def _usb_connection(
    args: argparse.Namespace,
    printer_class: type[LabelPrinter] | None = None,
    product_id: int | None = None,
    serial: str | None = None,
) -> ConnectionUSB:
    """Create a USB connection from CLI arguments and optional detection."""
    if isinstance(args.usb, str):
        vendor_id, uri_product_id, uri_serial = parse_usb_uri(args.usb)
        product_id = uri_product_id or product_id
        serial = args.serial or uri_serial or serial
        return ConnectionUSB(vendor_id=vendor_id, product_id=product_id, serial=serial)

    if product_id is None and printer_class is not None:
        product_id = printer_class.USB_PRODUCT_ID
    return ConnectionUSB(product_id=product_id, serial=args.serial or serial)


def _resolve_usb_printer(args: argparse.Namespace) -> tuple[type[LabelPrinter], int, str | None]:
    """Find the USB printer class/product/serial from arguments or enumeration."""
    if args.printer:
        printer_class = PRINTER_TYPES[args.printer]
        product_id = printer_class.USB_PRODUCT_ID
        serial = args.serial
        if isinstance(args.usb, str):
            _, uri_product_id, uri_serial = parse_usb_uri(args.usb)
            product_id = uri_product_id or product_id
            serial = args.serial or uri_serial
        return printer_class, product_id, serial

    if isinstance(args.usb, str):
        _, product_id, serial = parse_usb_uri(args.usb)
        if product_id is not None:
            printer_class = printer_class_for_usb_product_id(product_id)
            if printer_class is not None:
                return printer_class, product_id, args.serial or serial

    devices = discover_usb()
    if args.serial:
        devices = [device for device in devices if device["serial"] == args.serial]
    else:
        devices = [device for device in devices if device["usable"]]
    if not devices:
        name = f" with serial {args.serial}" if args.serial else ""
        raise RuntimeError(f"No supported Brother P-touch printer found on USB{name}.")
    if len(devices) > 1:
        names = ", ".join(
            f"{device['model'] or 'unknown'} ({device['serial'] or 'no serial'})"
            for device in devices
        )
        raise RuntimeError(
            f"Several USB printers found: {names}. Select one with --serial or --printer."
        )
    device = devices[0]
    product_id = int(str(device["product_id"]), 16)
    printer_class = printer_class_for_usb_product_id(product_id)
    if printer_class is None:
        raise RuntimeError(f"Unsupported USB printer (product ID 0x{product_id:04x}).")
    return printer_class, product_id, str(device["serial"]) if device["serial"] else None


def _make_connection(
    args: argparse.Namespace,
    printer_class: type[LabelPrinter] | None = None,
    product_id: int | None = None,
    serial: str | None = None,
) -> ConnectionNetwork | ConnectionUSB:
    """Create a connection from parsed command-line arguments."""
    if args.host:
        return ConnectionNetwork(args.host)
    return _usb_connection(args, printer_class, product_id, serial)


def _fetch_status(
    args: argparse.Namespace,
    printer_class: type[LabelPrinter] | None = None,
) -> tuple[PrinterStatus, LabelPrinter | None]:
    """Fetch printer status over the configured connection."""
    if args.host:
        return (
            query_status_snmp(
                args.host,
                community=args.snmp_community,
                version=args.snmp_version,
            ),
            None,
        )

    if printer_class is None:
        printer_class, product_id, serial = _resolve_usb_printer(args)
    else:
        product_id = printer_class.USB_PRODUCT_ID
        serial = args.serial
        if isinstance(args.usb, str):
            _, uri_product_id, uri_serial = parse_usb_uri(args.usb)
            product_id = uri_product_id or product_id
            serial = args.serial or uri_serial
    connection = _usb_connection(args, printer_class, product_id, serial)
    printer = printer_class(connection)
    return printer.get_status(
        community=args.snmp_community,
        snmp_version=args.snmp_version,
    ), printer


def _run_status(args: argparse.Namespace) -> int:
    """Query the printer status and print it."""
    try:
        status, _ = _fetch_status(args)
    except Exception as e:
        if args.json:
            print(
                json.dumps(
                    {
                        "connected": False,
                        "ready": False,
                        "message": str(e),
                        "ptouch_version": __version__,
                    },
                    indent=2,
                )
            )
            return 1
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print_status(status, as_json=args.json)
    return 0


def _run_list_connected(args: argparse.Namespace) -> int:
    """List USB printers."""
    try:
        printers = discover_usb()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(printers, indent=2))
        return 0
    if not printers:
        print("No Brother P-touch printers found on USB.")
        return 0
    for printer in printers:
        state = "" if printer["usable"] else " (unsupported)"
        print(
            f"{printer['model'] or 'unknown'}\tserial {printer['serial'] or 'unknown'}"
            f"\t({printer['product_id']}){state}"
        )
    return 0


def _run_scan(args: argparse.Namespace) -> int:
    """Scan the network for Brother printers."""
    cidr = args.scan if isinstance(args.scan, str) else None
    try:
        printers = scan_network(
            cidr,
            community=args.snmp_community,
            version=args.snmp_version,
            timeout=args.scan_timeout,
            workers=args.scan_workers,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(printers, indent=2))
        return 0
    if not printers:
        print("No Brother printers found.")
        return 0
    for printer in printers:
        media = _format_media_width(printer["media_width_mm"])
        state = "ready" if printer["ready"] else "not ready"
        print(
            f"{printer['host']}\t{printer['model'] or 'unknown'}"
            f"\tserial {printer['serial'] or 'unknown'}\t{media}\t{state}"
        )
    return 0


def main() -> int:
    """Run the command-line interface."""
    args = parse_args()

    if args.serial and not args.usb:
        print("Error: --serial is only valid with --usb", file=sys.stderr)
        return 1

    if args.list_connected:
        if args.status or args.scan is not None or args.host or args.usb:
            print("Error: --list-connected cannot be combined with other modes", file=sys.stderr)
            return 1
        return _run_list_connected(args)

    if args.scan is not None:
        if args.status or args.host or args.usb or args.text or args.image:
            print("Error: --scan cannot be combined with other modes", file=sys.stderr)
            return 1
        if args.scan_timeout <= 0 or args.scan_workers < 1:
            print("Error: --scan-timeout and --scan-workers must be positive", file=sys.stderr)
            return 1
        return _run_scan(args)

    if args.status:
        if args.text or args.image:
            print("Error: Cannot combine --status with text or --image", file=sys.stderr)
            return 1
        if args.tape_width is not None or args.tube_width is not None:
            print(
                "Error: Cannot combine --status with --tape-width or --tube-width",
                file=sys.stderr,
            )
            return 1
        if not args.host and args.usb is None:
            print("Error: --status needs --host or --usb", file=sys.stderr)
            return 1
        return _run_status(args)

    if args.json:
        print("Error: --json requires --status, --scan or --list-connected", file=sys.stderr)
        return 1

    if not args.host and args.usb is None:
        print(
            "Error: Specify --host or --usb (or use --status/--scan/--list-connected)",
            file=sys.stderr,
        )
        return 1

    # Validate content
    if args.image and args.text:
        print("Error: Cannot specify both text and --image", file=sys.stderr)
        return 1
    if not args.image and not args.text:
        print("Error: Must specify either text or --image", file=sys.stderr)
        return 1
    if args.copies < 1:
        print("Error: --copies must be at least 1", file=sys.stderr)
        return 1
    if args.margin is not None and args.margin < 0:
        print("Error: --margin must not be negative", file=sys.stderr)
        return 1

    # Resolve printer and media, auto-detecting what was not given.
    printer_class = PRINTER_TYPES[args.printer] if args.printer else None
    media_class: type[Tape] | None = None
    if args.tape_width is not None:
        media_class = TAPE_WIDTHS[args.tape_width]
    elif args.tube_width is not None:
        media_class = TUBE_WIDTHS[args.tube_width]

    status: PrinterStatus | None = None
    usb_printer: LabelPrinter | None = None
    try:
        if printer_class is None or media_class is None:
            status, usb_printer = _fetch_status(args, printer_class)
        if printer_class is None:
            if status is None or status.model is None:
                print("Error: Could not detect the printer model; use --printer", file=sys.stderr)
                return 1
            printer_class = printer_class_for_model(status.model)
            if printer_class is None:
                print(
                    f"Error: Unsupported printer model {status.model!r}; use --printer",
                    file=sys.stderr,
                )
                return 1
        if media_class is None:
            if status is None or status.media_width_mm is None:
                print(
                    "Error: Could not detect the loaded media; use --tape-width or --tube-width",
                    file=sys.stderr,
                )
                return 1
            media_class = resolve_media_class(status.media_width_mm, status.media_type_code)
            if media_class is None:
                width = status.media_width_mm
                print(
                    f"Error: Unsupported media width {width:g} mm "
                    f"({status.media_type_label}); use --tape-width or --tube-width",
                    file=sys.stderr,
                )
                return 1

        if usb_printer is None:
            connection = _make_connection(args, printer_class)
            printer = printer_class(connection)
        else:
            printer = usb_printer
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.printer == "P910BT" and args.tube_width is not None:
        print("Error: PT-P910BT does not support heat shrink tubes", file=sys.stderr)
        return 1

    use_compression = not args.no_compression
    printer.use_compression = use_compression
    printer.high_resolution = args.high_resolution

    # Create label(s)
    if args.image:
        image = Image.open(args.image)
        labels = [Label(image, media_class)]
    else:
        h_align = ALIGN_HORIZONTAL.get(args.align[0].lower())
        v_align = ALIGN_VERTICAL.get(args.align[1].lower())
        if h_align is None:
            print(
                f"Error: Invalid horizontal alignment '{args.align[0]}'. Use: left, center, right",
                file=sys.stderr,
            )
            return 1
        if v_align is None:
            print(
                f"Error: Invalid vertical alignment '{args.align[1]}'. Use: top, center, bottom",
                file=sys.stderr,
            )
            return 1
        align = h_align | v_align

        font: str | ImageFont.FreeTypeFont
        if args.font:
            font = args.font
        else:
            try:
                default_font = ImageFont.load_default()
                if not hasattr(default_font, "font_variant") or not isinstance(
                    default_font, ImageFont.FreeTypeFont
                ):
                    print(
                        "Error: PIL default font is not scalable. "
                        "Please upgrade Pillow to 10.1+ or provide --font",
                        file=sys.stderr,
                    )
                    return 1
                font = default_font
            except Exception as e:
                print(f"Error: Could not load default font: {e}", file=sys.stderr)
                print("Please provide --font with a path to a TrueType font", file=sys.stderr)
                return 1

        margin_mm = args.margin if args.margin is not None else LabelPrinter.DEFAULT_MARGIN_MM
        fixed_width_mm = None
        if args.width is not None:
            fixed_width_mm = args.width - (2 * margin_mm)
            if fixed_width_mm <= 0:
                print(
                    f"Error: --width must be greater than 2x margin ({2 * margin_mm}mm)",
                    file=sys.stderr,
                )
                return 1

        auto_size = args.font_size is None
        labels = create_text_labels(
            args.text,
            media_class,
            font=font,
            align=align,
            font_size=args.font_size,
            fixed_width_mm=fixed_width_mm,
            auto_size=auto_size,
        )

    if args.copies > 1:
        labels = labels * args.copies

    send_margin = not args.no_margin and (args.margin is None or args.margin > 0)
    margin_mm = args.margin if send_margin else None

    num_labels = len(labels)
    use_half_cut = not args.full_cut and not args.chain and printer_class.SUPPORTS_HALF_CUT
    conn_type = "network" if args.host else "USB"
    print(f"Printing {num_labels} label(s) to {printer_class.__name__} via {conn_type}...")
    media_label = "Tube" if media_class.__name__.startswith("HeatShrink") else "Tape"
    print(
        f"{media_label}: {media_class.width_mm:g}mm, High-res: {args.high_resolution}, "
        f"Compression: {use_compression}"
    )

    try:
        if args.precut:
            printer.precut(labels[0].tape)
        if num_labels == 1:
            printer.print(
                labels[0],
                margin_mm=margin_mm,
                high_resolution=args.high_resolution,
                chain=args.chain,
                mirror=args.mirror,
                send_margin=send_margin,
                vertical_offset=args.vertical_offset,
            )
        else:
            printer.print_multi(
                labels,
                margin_mm=margin_mm,
                high_resolution=args.high_resolution,
                half_cut=use_half_cut,
                chain=args.chain,
                mirror=args.mirror,
                send_margin=send_margin,
                vertical_offset=args.vertical_offset,
            )
        print("Done.")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
