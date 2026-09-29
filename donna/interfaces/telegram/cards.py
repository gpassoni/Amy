"""Rendering proposals for Telegram.

Kept separate from the handlers so the same text can be reused by the notifier, by a callback
that edits a message in place, and later by the web UI.

Text is plain, not Markdown. Telegram's parsers choke on unescaped characters that appear
constantly in real email subjects — underscores, asterisks, square brackets — and a message
that fails to send is worse than one without bold text.
"""

from __future__ import annotations

from donna.pipeline import resolve
from donna.store import repo
from donna.timeutil import format_it, format_range_it, parse_iso

# Telegram's hard limit is 4096; leave room for the keyboard and any appended note.
MAX_LEN = 3800


def proposal_card(row) -> str:
    """The message a proposal arrives as."""
    payload = repo.proposal_payload(row)
    start, end = parse_iso(payload.get("start_ts")), parse_iso(payload.get("end_ts"))
    title = payload.get("title") or "(senza titolo)"

    if row["kind"] in ("calendar_move", "calendar_delete"):
        # These come from a request in chat, and the card has to say what will change.
        return f"📌 Da confermare\n\n{resolve.describe(row)}"[:MAX_LEN]

    header = (
        "📌 Ho trovato un impegno in una email."
        if row["source_type"] == "email"
        else "📌 Da confermare."
    )
    lines = [header, "", title]

    if start is not None:
        when = (
            f"{format_it(start, with_time=False)} (tutto il giorno)"
            if payload.get("all_day")
            else format_range_it(start, end)
        )
        lines.append(f"🕐 {when}")
    if payload.get("location"):
        lines.append(f"📍 {payload['location']}")

    source = repo.get_email(row["source_id"]) if row["source_type"] == "email" else None
    if source is not None:
        sender = source["from_name"] or source["from_addr"]
        lines.append(f"✉️ da {sender}")
        lines.append(f"   {(source['subject'] or '')[:80]}")

    confidence = row["confidence"] or 0
    # Surfaced rather than hidden: a low-confidence proposal deserves a closer look, and
    # saying so is more honest than presenting every guess with equal certainty.
    if confidence < 0.7:
        lines.append("")
        lines.append(f"⚠️ Non sono sicurissima (confidenza {confidence:.0%}).")

    return "\n".join(lines)[:MAX_LEN]


def why_card(row) -> str:
    """The explanation behind a proposal: the evidence, then the mechanics."""
    payload = repo.proposal_payload(row)
    lines = [f"🔎 Perché ti ho proposto #{row['id']}", ""]

    if row["evidence_quote"]:
        lines.append(f'Dall\'email: "{row["evidence_quote"]}"')
        lines.append("")
    if row["reasoning"]:
        lines.append(row["reasoning"])
        lines.append("")

    phrase = payload.get("date_phrase")
    source = payload.get("date_source")
    if phrase:
        lines.append(f'Data dalla frase: "{phrase}"')
    if source:
        explanation = {
            "agreed": "confermata due volte, dal testo e dalla mia stima",
            "phrase": "ricavata dal testo dell'email",
            "model": "stimata da me, senza una frase esplicita",
        }.get(source, source)
        lines.append(f"Come l'ho ricavata: {explanation}")

    lines.append(f"Confidenza: {(row['confidence'] or 0):.0%}")
    return "\n".join(lines)[:MAX_LEN]


def resolved_card(row, resolution: resolve.Resolution) -> str:
    """What the card becomes once the user has decided, so the transcript stays readable."""
    payload = repo.proposal_payload(row)
    title = payload.get("title") or "(senza titolo)"
    start, end = parse_iso(payload.get("start_ts")), parse_iso(payload.get("end_ts"))
    when = ""
    if start is not None:
        when = (
            format_it(start, with_time=False)
            if payload.get("all_day")
            else format_range_it(start, end)
        )

    if resolution.state == "accepted":
        if row["kind"] in ("calendar_move", "calendar_delete"):
            return f"✅ Fatto: {resolve.describe(row)}"
        return f"✅ In calendario: {title}\n🕐 {when}"
    if resolution.state == "rejected":
        return f"🗑 Scartata: {title}"
    if resolution.state == "expired":
        return f"⏳ Scaduta senza risposta: {title}"
    return f"{title} — {resolution.state}"


def split(text: str, limit: int = 4000) -> list[str]:
    """Split a long reply on line boundaries.

    Carried over from v1's splitter, which handled this correctly: break at the last newline
    before the limit, fall back to a space, and only then cut mid-word.
    """
    if not text:
        return ["(nessuna risposta)"]
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = text.rfind(" ", 0, limit)
        if cut == -1:
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    return [p for p in parts if p]
