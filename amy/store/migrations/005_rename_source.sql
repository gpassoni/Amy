-- The project was renamed from Donna to Amy. Events created through an accepted proposal
-- carry the assistant's name as their provenance, so existing rows follow the rename.
UPDATE events SET source = 'amy' WHERE source = 'donna';
