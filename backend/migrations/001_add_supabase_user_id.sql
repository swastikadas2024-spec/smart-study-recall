ALTER TABLE public.profiles
ADD COLUMN IF NOT EXISTS supabase_user_id UUID;

CREATE UNIQUE INDEX IF NOT EXISTS
uq_profiles_supabase_user_id
ON public.profiles (supabase_user_id)
WHERE supabase_user_id IS NOT NULL;