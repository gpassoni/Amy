"""Gmail -> local mirror.

Incremental via historyId. The first run backfills a bounded window; every run after that
replays only what changed, which is what makes a 5-minute schedule cheap enough to leave
running.

The ordering here matters: the history cursor is captured *before* the backfill listing, so
a message that arrives mid-backfill is replayed by the next incremental pass instead of
being lost in the gap.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from googleapiclient.errors import HttpError

from donna.config import get_settings
from donna.google import gmail
from donna.store import repo
from donna.sync.base import SyncResult
from donna.timeutil import iso_utc, parse_email_date

logger = logging.getLogger(__name__)

RESOURCE = "gmail"
# Backfill completion is tracked separately from the change cursor, because the two can
# legitimately disagree: a backfill interrupted by a rate limit leaves messages unfetched
# that the forward-only change feed will never mention again. Until this reads "done", every
# run takes the full path — which is cheap, since already-stored ids are skipped.
BACKFILL_RESOURCE = "gmail_backfill"
BACKFILL_DONE = "done"


def _received_at(message: gmail.Message) -> str:
    """Best available receipt time, in storage form.

    internalDate is Gmail's own record of when it accepted the message and is always
    present and always sane. The Date: header is sender-written and frequently is not, so
    it is only a fallback — and if both fail we still need *a* timestamp, because
    received_at is NOT NULL and ordering depends on it.
    """
    if message.internal_date:
        try:
            epoch_ms = int(message.internal_date)
            return iso_utc(datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc))
        except (ValueError, OverflowError, OSError):
            logger.debug("Unusable internalDate %r on %s", message.internal_date, message.id)

    parsed = parse_email_date(message.date_header)
    if parsed:
        return iso_utc(parsed)

    logger.warning("No usable date on message %s; falling back to now", message.id)
    return iso_utc(datetime.now(timezone.utc))


def _store(message: gmail.Message) -> bool:
    return repo.upsert_email(
        id=message.id,
        thread_id=message.thread_id,
        from_addr=message.from_addr,
        from_name=message.from_name,
        to_addrs=message.to_addrs,
        subject=message.subject,
        snippet=message.snippet,
        body=message.body,
        received_at=_received_at(message),
        label_ids=message.label_ids,
        is_unread=message.is_unread,
    )


def sync_gmail(*, full: bool = False, limit: int | None = None) -> SyncResult:
    settings = get_settings()
    started = time.perf_counter()
    cursor = repo.get_cursor(RESOURCE)
    result = SyncResult(resource=RESOURCE)

    backfill_done = repo.get_cursor(BACKFILL_RESOURCE) == BACKFILL_DONE

    try:
        if full or not cursor or not backfill_done:
            result.mode = "full"
            # Capture the cursor first — see the module docstring.
            new_cursor = gmail.current_history_id()
            listed = gmail.list_recent_ids(
                days=settings.gmail_sync_window_days,
                limit=limit or settings.gmail_max_per_sync,
            )
            # Only fetch what we do not already hold, so an interrupted backfill resumes
            # instead of re-downloading everything and hitting the quota again.
            already = repo.known_email_ids(listed)
            ids = [i for i in listed if i not in already]
            result.skipped = len(already)
            logger.info(
                "Gmail backfill: %d messaggi negli ultimi %d giorni, %d già presenti, %d da scaricare",
                len(listed),
                settings.gmail_sync_window_days,
                len(already),
                len(ids),
            )
        else:
            try:
                touched, removed, new_cursor = gmail.changed_ids_since(cursor)
            except gmail.HistoryExpired as exc:
                logger.warning("%s — falling back to a full sync", exc)
                return sync_gmail(full=True, limit=limit)

            ids = list(touched)
            if limit:
                ids = ids[:limit]
            result.deleted = repo.delete_emails(removed)

        report = gmail.fetch_messages(ids)
        for message in report.messages:
            if _store(message):
                result.created += 1
            else:
                result.updated += 1

        # Messages Gmail no longer has are genuinely gone, not a failure.
        result.skipped += len(report.missing)
        result.duration_ms = int((time.perf_counter() - started) * 1000)

        if not report.complete:
            # Do NOT advance the cursor on a partial pass. Advancing it would move the
            # change feed past messages we never stored, and the incremental path only
            # moves forward — they would be lost permanently. Leaving the cursor where it
            # is means the next run retries the gap.
            result.error = (
                f"{len(report.failed)} messaggi non recuperati (probabile limite di quota); "
                "cursore non avanzato, il prossimo sync riprende da qui"
            )
            repo.record_sync(RESOURCE, error=result.error, items=result.created)
            logger.warning("Gmail sync incompleto: %s", result.error)
            return result

        repo.record_sync(RESOURCE, cursor=new_cursor, items=result.created)
        if result.mode == "full":
            repo.record_sync(BACKFILL_RESOURCE, cursor=BACKFILL_DONE)
        logger.info("Gmail sync: %s", result.summary())
        return result

    except (HttpError, OSError, RuntimeError) as exc:
        result.error = str(exc)
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        repo.record_sync(RESOURCE, error=str(exc))
        logger.error("Gmail sync failed: %s", exc, exc_info=True)
        return result
