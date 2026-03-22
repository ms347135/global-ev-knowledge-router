from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[list[str]]) -> bytes:
    objects: list[bytes] = []

    def add_object(data: str) -> int:
        objects.append(data.encode("latin-1", errors="replace"))
        return len(objects)

    font_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids = []
    content_ids = []
    pages_id_placeholder = len(objects) + 2

    for lines in pages:
        content_lines = ["BT", "/F1 11 Tf", "50 760 Td", "14 TL"]
        first = True
        for line in lines:
            safe = escape_pdf_text(line)
            if first:
                content_lines.append(f"({safe}) Tj")
                first = False
            else:
                content_lines.append(f"T* ({safe}) Tj")
        content_lines.append("ET")
        stream = "\n".join(content_lines)
        content_id = add_object(f"<< /Length {len(stream.encode('latin-1', errors='replace'))} >>\nstream\n{stream}\nendstream")
        content_ids.append(content_id)
        page_id = add_object(
            f"<< /Type /Page /Parent {pages_id_placeholder} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        )
        page_ids.append(page_id)

    pages_kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    pages_id = add_object(f"<< /Type /Pages /Kids [{pages_kids}] /Count {len(page_ids)} >>")
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]

    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    return bytes(pdf)


DOCUMENTS = {
    "vehicle_specs/seal_global_specs_demo.pdf": [
        [
            "Seal Global Specs Demo",
            "Vehicle model: Seal",
            "Document type: vehicle_specs",
            "Market: global",
            "Battery options: 61.4 kWh and 82.5 kWh.",
            "Estimated WLTP range depends on trim and market configuration.",
            "DC charging capability varies by battery and trim package.",
            "Common support question: compare charging speed against local charger limits.",
        ],
        [
            "Seal Charging and Performance Notes",
            "AC charging is intended for overnight or destination charging use.",
            "DC charging performance depends on battery temperature and charger capacity.",
            "Battery preconditioning can improve fast charging performance when supported.",
            "Published figures should be interpreted together with local market documentation.",
        ],
    ],
    "charging_guides/dolphin_thailand_home_charging_demo.pdf": [
        [
            "Dolphin Thailand Home Charging Guide Demo",
            "Vehicle model: Dolphin",
            "Document type: charging_guide",
            "Market: thailand",
            "Use certified AC charging equipment appropriate for local electrical standards.",
            "Home charger installation should be completed by qualified professionals.",
            "Check charger rating, cable condition, and connector cleanliness before use.",
        ],
        [
            "Charging Best Practices",
            "Avoid repeated high-state-of-charge parking for long storage periods.",
            "If available, battery preconditioning may support improved DC charging consistency.",
            "If charging is interrupted, inspect connector seating and charger status messages first.",
            "Consult local installation notes for property, utility, and safety requirements.",
        ],
    ],
    "warranty_terms/seal_brazil_warranty_demo.pdf": [
        [
            "Seal Brazil Warranty Demo",
            "Vehicle model: Seal",
            "Document type: warranty_terms",
            "Market: brazil",
            "Battery and vehicle warranty coverage differ by component category.",
            "Warranty claims may require documented service history and approved maintenance practices.",
            "Consumables and misuse cases are commonly excluded from standard warranty coverage.",
        ],
        [
            "Warranty Support Notes",
            "Customers should be advised to retain charging equipment and service invoices when relevant.",
            "Warranty interpretation may vary with local terms, campaign notices, and model year updates.",
            "Use local support documentation when explaining claim steps and exclusions.",
        ],
    ],
    "owner_manuals/atto3_dashboard_warnings_demo.pdf": [
        [
            "Atto 3 Dashboard Warnings Demo",
            "Vehicle model: Atto 3",
            "Document type: owner_manual",
            "Market: global",
            "Dashboard warnings should be interpreted together with text messages and vehicle behavior.",
            "A tire pressure warning typically suggests checking each tire and driving condition promptly.",
            "High voltage or braking related warnings require cautious escalation and service review.",
        ],
        [
            "Driver Guidance Notes",
            "Do not advise users to open high-voltage components.",
            "For persistent critical warnings, recommend limiting vehicle use and contacting authorized service.",
            "For informational warnings, provide safe user checks before escalation.",
        ],
    ],
    "service_faqs/global_service_troubleshooting_demo.pdf": [
        [
            "Global Service FAQ and Troubleshooting Demo",
            "Document type: service_faq",
            "Market: global",
            "Question: charging stopped at 82 percent.",
            "Possible causes include charger-side limits, battery temperature management, or user settings.",
            "Safe first checks include connector seating, charger status screen, and visible cable issues.",
        ],
        [
            "Escalation Guidance",
            "If repeated charging interruptions occur across multiple chargers, recommend service review.",
            "If unusual smell, heat, or warning icons appear, stop charging and seek professional support.",
            "Do not instruct customers to disassemble connectors or battery components.",
        ],
    ],
    "market_policy/hungary_home_charger_installation_demo.pdf": [
        [
            "Hungary Home Charger Installation Policy Demo",
            "Document type: market_policy",
            "Market: hungary",
            "Local installation communication should reference qualified electricians and applicable property rules.",
            "Customer guidance should clearly separate product guidance from local electrical compliance requirements.",
            "Installation timelines may depend on site inspection, utility conditions, and property approvals.",
        ],
        [
            "Regional Communication Notes",
            "Support teams should avoid overpromising installation speed without site verification.",
            "Charging guidance should be paired with local safety and compliance notes when available.",
            "For regional uncertainty, official local installer guidance should be checked.",
        ],
    ],
}


def main() -> None:
    for relative_path, pages in DOCUMENTS.items():
        destination = BASE_DIR / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(build_pdf(pages))
        print(f"generated {destination}")


if __name__ == "__main__":
    main()
