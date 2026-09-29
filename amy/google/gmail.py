"""Gmail access.

The MIME body walk and the label helpers are carried over from v1's services/gmail_api.py
— that code was already correct and handles the awkward nested-multipart cases.

What is new is incremental sync. v1 re-queried `is:unread newer_than:3d` and re-read every
message every time, which is both slow and unable to notice that a message was read or
relabelled elsewhere. Gmail's history API gives us a change feed instead.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from email.utils import parseaddr
from typing import Any

from bs4 import BeautifulSoup
from googleapiclient.errors import HttpError

from amy.google import retry
from amy.google.auth import gmail as gmail_service

logger = logging.getLogger(__name__)

# Amy's own labels, kept identical to v1 so existing labels in the mailbox still apply.
LABEL_NAMES = {
    "importante": "🔴 Importanti",
    "da_leggere": "🟡 Da Leggere",
    "inutile": "⚪ Inutili",
}

# Body text kept per message. Enough for classification and extraction without bloating
# the database or the model's context.
BODY_LIMIT = 4000


class HistoryExpired(RuntimeError):
    """The stored historyId is too old; Gmail can no longer replay from it."""


@dataclass(slots=True)
class Message:
    id: str
    thread_id: str
    from_addr: str
    from_name: str
    to_addrs: str
    subject: str
    snippet: str
    body: str
    date_header: str
    # Epoch milliseconds, as Gmail reports it. Preferred over the Date: header, which is
    # written by the sender and is routinely wrong, missing, or in an unparseable dialect.
    internal_date: str
    label_ids: list[str] = field(default_factory=list)

    @property
    def is_unread(self) -> bool:
        return "UNREAD" in self.label_ids


# ---------------------------------------------------------------- body extraction
def extract_body(payload: dict[str, Any]) -> str:
    """Walk a Gmail payload for readable text.

    Prefers text/plain, falls back to stripped HTML, and recurses into nested multiparts
    (multipart/mixed wrapping multipart/alternative is common and trips naive versions).
    """
    parts = payload.get("parts") or []

    if not parts:
        data = (payload.get("body") or {}).get("data") or ""
        if not data:
            return ""
        decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        if payload.get("mimeType") == "text/html":
            return BeautifulSoup(decoded, "html.parser").get_text(separator="\n", strip=True)
        return decoded

    for part in parts:
        if part.get("mimeType") == "text/plain" and (part.get("body") or {}).get("data"):
            return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")

    for part in parts:
        if part.get("mimeType") == "text/html" and (part.get("body") or {}).get("data"):
            html = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
            return BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)

    for part in parts:
        if "parts" in part:
            nested = extract_body(part)
            if nested:
                return nested

    return ""


def _header(headers: list[dict[str, str]], name: str, default: str = "") -> str:
    lowered = name.lower()
    return next((h["value"] for h in headers if h.get("name", "").lower() == lowered), default)


def _to_message(raw: dict[str, Any]) -> Message:
    payload = raw.get("payload") or {}
    headers = payload.get("headers") or []
    sender = _header(headers, "from", "Sconosciuto")
    name, addr = parseaddr(sender)
    body = extract_body(payload)

    return Message(
        id=raw["id"],
        thread_id=raw.get("threadId", ""),
        from_addr=addr or sender,
        from_name=name or (addr.split("@")[0] if addr else sender),
        to_addrs=_header(headers, "to"),
        subject=_header(headers, "subject", "(nessun oggetto)"),
        snippet=raw.get("snippet", ""),
        body=body[:BODY_LIMIT],
        date_header=_header(headers, "date"),
        internal_date=str(raw.get("internalDate") or ""),
        label_ids=raw.get("labelIds") or [],
    )


# ---------------------------------------------------------------- reads
def current_history_id() -> str:
    """The mailbox's current historyId.

    Captured *before* a full sync, so anything arriving during the sync is replayed by the
    next incremental pass rather than being skipped.
    """
    profile = retry.execute(
        gmail_service().users().getProfile(userId="me"), label="gmail getProfile"
    )
    return str(profile["historyId"])


def get_message(message_id: str) -> Message | None:
    try:
        raw = gmail_service().users().messages().get(userId="me", id=message_id).execute()
    except HttpError as exc:
        if exc.resp.status == 404:
            return None  # deleted between being listed and being fetched
        raise
    return _to_message(raw)


def list_recent_ids(days: int, limit: int) -> list[str]:
    """Message ids from the last `days`, newest first. Used for the initial backfill."""
    service = gmail_service()
    ids: list[str] = []
    page_token: str | None = None

    while len(ids) < limit:
        response = (
            service.users()
            .messages()
            .list(
                userId="me",
                q=f"newer_than:{days}d",
                maxResults=min(500, limit - len(ids)),
                pageToken=page_token,
            )
            .execute()
        )
        ids.extend(m["id"] for m in response.get("messages") or [])
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return ids[:limit]


@dataclass(slots=True)
class HistoryChanges:
    """What the change feed reported, split by what each change actually requires.

    The distinction between `added` and `relabelled` exists because conflating them created a
    feedback loop: triage applies a Gmail label to every message it classifies, each label
    write becomes a history entry, and treating those entries as "this message changed" made
    the next sync re-download the entire mailbox. Labelling 154 messages caused 154
    re-fetches and blew the API quota.

    A label change cannot alter a subject or a body, and the history entry already carries
    the label ids — so it needs no fetch at all.
    """

    added: set[str] = field(default_factory=set)
    deleted: set[str] = field(default_factory=set)
    # message id -> (labels added, labels removed)
    relabelled: dict[str, tuple[set[str], set[str]]] = field(default_factory=dict)
    cursor: str = ""

    def note_labels(self, message_id: str, added: list[str], removed: list[str]) -> None:
        current_added, current_removed = self.relabelled.setdefault(message_id, (set(), set()))
        current_added.update(added)
        current_removed.update(removed)


def replay_history(history_id: str) -> HistoryChanges:
    """Replay the change feed from `history_id`.

    Raises HistoryExpired when the cursor has aged out — Gmail keeps roughly a week of
    history, so a laptop that was off for a fortnight needs a full resync.
    """
    service = gmail_service()
    changes = HistoryChanges(cursor=history_id)
    latest = history_id
    page_token: str | None = None

    while True:
        try:
            response = (
                service.users()
                .history()
                .list(
                    userId="me",
                    startHistoryId=history_id,
                    historyTypes=["messageAdded", "messageDeleted", "labelAdded", "labelRemoved"],
                    pageToken=page_token,
                )
                .execute()
            )
        except HttpError as exc:
            if exc.resp.status == 404:
                raise HistoryExpired(
                    f"historyId {history_id} troppo vecchio per Gmail; serve un sync completo"
                ) from exc
            raise

        for entry in response.get("history") or []:
            latest = str(entry.get("id", latest))
            for added in entry.get("messagesAdded") or []:
                changes.added.add(added["message"]["id"])
            for removed in entry.get("messagesDeleted") or []:
                changes.deleted.add(removed["message"]["id"])
            for item in entry.get("labelsAdded") or []:
                changes.note_labels(item["message"]["id"], item.get("labelIds") or [], [])
            for item in entry.get("labelsRemoved") or []:
                changes.note_labels(item["message"]["id"], [], item.get("labelIds") or [])

        page_token = response.get("nextPageToken")
        if not page_token:
            latest = str(response.get("historyId", latest))
            break

    changes.added -= changes.deleted
    for message_id in changes.deleted:
        changes.relabelled.pop(message_id, None)
    # A brand-new message is fetched in full anyway; its label deltas are redundant.
    for message_id in changes.added:
        changes.relabelled.pop(message_id, None)
    changes.cursor = latest
    return changes


# Gmail bills messages.get at 5 quota units against a 250-unit-per-second budget, so 50
# messages per second is the ceiling. A batch of 25 plus spacing keeps a backfill
# comfortably under it while still cutting round trips by 25x.
BATCH_SIZE = 25
_limiter = retry.RateLimiter(min_interval=0.6)


@dataclass(slots=True)
class FetchReport:
    """Outcome of a fetch pass. `failed` is what makes partial progress visible: a caller
    that advances its sync cursor while messages are still missing loses them forever."""

    messages: list[Message] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # deleted server-side; not an error
    failed: list[str] = field(default_factory=list)  # could not be fetched; retry later

    @property
    def complete(self) -> bool:
        return not self.failed


def fetch_messages(ids: list[str]) -> FetchReport:
    """Fetch messages in batches, tolerating individual failures.

    One unparseable or vanished message must not abort the pass, but the ids that failed
    are reported rather than swallowed, so the caller can decline to advance its cursor.
    """
    report = FetchReport()
    if not ids:
        return report

    service = gmail_service()

    for offset in range(0, len(ids), BATCH_SIZE):
        chunk = ids[offset : offset + BATCH_SIZE]
        collected: dict[str, dict[str, Any]] = {}
        errors: dict[str, Exception] = {}

        def _callback(request_id: str, response: Any, exception: Exception | None) -> None:
            if exception is not None:
                errors[request_id] = exception  # noqa: B023 — batch runs within this iteration
            else:
                collected[request_id] = response  # noqa: B023

        batch = service.new_batch_http_request()
        for message_id in chunk:
            batch.add(
                service.users().messages().get(userId="me", id=message_id),
                request_id=message_id,
                callback=_callback,
            )

        _limiter.wait()
        try:
            retry.with_backoff(batch.execute, label=f"gmail batch of {len(chunk)}")
        except HttpError as exc:
            # The whole batch was rejected even after backoff. Report the chunk as failed
            # and keep going: a later chunk may well succeed, and the caller decides.
            logger.warning("Gmail batch of %d failed: %s", len(chunk), exc)
            report.failed.extend(chunk)
            continue

        for message_id in chunk:
            if message_id in collected:
                try:
                    report.messages.append(_to_message(collected[message_id]))
                except (KeyError, ValueError, TypeError):
                    logger.warning("Could not parse message %s", message_id, exc_info=True)
                    report.failed.append(message_id)
            else:
                exc = errors.get(message_id)
                status = getattr(getattr(exc, "resp", None), "status", None)
                if status == 404:
                    report.missing.append(message_id)
                else:
                    report.failed.append(message_id)

        if report.failed:
            # One aggregate line, not a traceback per message: a quota failure across a
            # backfill otherwise produced tens of kilobytes of identical stack traces.
            logger.warning(
                "Gmail fetch: %d/%d recuperati, %d da riprovare",
                len(report.messages),
                len(ids),
                len(report.failed),
            )

    return report


# ---------------------------------------------------------------- labels
_label_cache: dict[str, str] | None = None


def ensure_labels(*, refresh: bool = False) -> dict[str, str]:
    """Map category key -> Gmail label id, creating any missing label. Cached per process."""
    global _label_cache
    if _label_cache is not None and not refresh:
        return _label_cache

    service = gmail_service()
    existing = {
        label["name"]: label["id"]
        for label in service.users().labels().list(userId="me").execute().get("labels") or []
    }

    resolved: dict[str, str] = {}
    for key, name in LABEL_NAMES.items():
        if name in existing:
            resolved[key] = existing[name]
            continue
        logger.info("Creating missing Gmail label %s", name)
        created = (
            service.users()
            .labels()
            .create(
                userId="me",
                body={
                    "name": name,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show",
                },
            )
            .execute()
        )
        resolved[key] = created["id"]

    _label_cache = resolved
    return resolved


def apply_category_label(message_id: str, category: str) -> bool:
    """Apply Amy's label for `category`, removing the other two.

    Removing the others matters on reclassification: without it a message that moves from
    inutile to importante ends up wearing both labels.
    """
    labels = ensure_labels()
    target = labels.get(category)
    if not target:
        logger.error("No Gmail label configured for category %r", category)
        return False

    remove = [lid for key, lid in labels.items() if key != category]
    try:
        gmail_service().users().messages().modify(
            userId="me",
            id=message_id,
            body={"addLabelIds": [target], "removeLabelIds": remove},
        ).execute()
    except HttpError:
        logger.warning("Could not label message %s as %s", message_id, category, exc_info=True)
        return False
    return True
