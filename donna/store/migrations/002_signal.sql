-- The raw perception label from the classifier, kept alongside the derived category.
-- Storing both means the importance policy (schemas.SIGNAL_CATEGORY) can be changed
-- and the whole mailbox recategorised without re-running a single model call.
ALTER TABLE emails ADD COLUMN category_signal TEXT;
CREATE INDEX idx_emails_signal ON emails(category_signal);
