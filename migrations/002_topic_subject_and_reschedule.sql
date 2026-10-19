-- Run once in the Supabase SQL editor (Database > SQL Editor > New query).
ALTER TABLE public.topics
ADD COLUMN IF NOT EXISTS subject VARCHAR(60);

ALTER TABLE public.revision_schedules
ADD COLUMN IF NOT EXISTS rescheduled_count INTEGER NOT NULL DEFAULT 0;
