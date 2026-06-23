-- ROSClaw v1.0 Feedback & Telemetry schema
-- Create all tables, indexes, and RLS policies for product telemetry and rich feedback.

-- -------------------------------------------------
-- telemetry_installations
-- -------------------------------------------------
create table if not exists public.telemetry_installations (
  id uuid default gen_random_uuid() primary key,
  anonymous_installation_id text unique not null,
  first_seen_at timestamptz not null default now(),
  last_seen_at timestamptz not null default now(),
  first_rosclaw_version text,
  latest_rosclaw_version text,
  os_family text,
  arch text,
  python_major_minor text,
  install_channel text,
  deployment_mode text,
  telemetry_status text not null default 'enabled'
);

create index if not exists idx_telemetry_installations_last_seen
  on public.telemetry_installations(last_seen_at);

create index if not exists idx_telemetry_installations_version
  on public.telemetry_installations(latest_rosclaw_version);

create index if not exists idx_telemetry_installations_os
  on public.telemetry_installations(os_family);

-- -------------------------------------------------
-- telemetry_events
-- -------------------------------------------------
create table if not exists public.telemetry_events (
  id uuid default gen_random_uuid() primary key,
  anonymous_installation_id text not null,
  schema_version text not null,
  event_type text not null,
  command_name text,
  command_status text,
  module_name text,
  rosclaw_version text,
  cli_version text,
  os_family text,
  arch text,
  python_major_minor text,
  install_channel text,
  deployment_mode text,
  duration_bucket text,
  error_class_bucket text,
  created_at timestamptz not null,
  received_at timestamptz not null default now(),
  payload jsonb not null default '{}'::jsonb
);

create index if not exists idx_telemetry_events_installation
  on public.telemetry_events(anonymous_installation_id);

create index if not exists idx_telemetry_events_type
  on public.telemetry_events(event_type);

create index if not exists idx_telemetry_events_command
  on public.telemetry_events(command_name);

create index if not exists idx_telemetry_events_module
  on public.telemetry_events(module_name);

create index if not exists idx_telemetry_events_received_at
  on public.telemetry_events(received_at);

-- -------------------------------------------------
-- telemetry_daily_aggregates
-- -------------------------------------------------
create table if not exists public.telemetry_daily_aggregates (
  id uuid default gen_random_uuid() primary key,
  day date not null,
  metric_name text not null,
  dimension jsonb not null default '{}'::jsonb,
  value numeric not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique(day, metric_name, dimension)
);

create index if not exists idx_telemetry_daily_metric
  on public.telemetry_daily_aggregates(metric_name);

create index if not exists idx_telemetry_daily_day
  on public.telemetry_daily_aggregates(day);

-- -------------------------------------------------
-- feedback_batches
-- -------------------------------------------------
create table if not exists public.feedback_batches (
  id uuid default gen_random_uuid() primary key,
  anonymous_installation_id text not null,
  schema_version text not null,
  event_count integer not null default 0,
  attachment_count integer not null default 0,
  redacted boolean not null default true,
  client_version text,
  created_at timestamptz not null default now(),
  received_at timestamptz not null default now(),
  status text not null default 'received',
  privacy_level text not null default 'L0',
  redaction_report jsonb not null default '{}'::jsonb
);

-- -------------------------------------------------
-- feedback_events
-- -------------------------------------------------
create table if not exists public.feedback_events (
  id uuid default gen_random_uuid() primary key,
  batch_id uuid references public.feedback_batches(id) on delete cascade,
  event_id text not null,
  anonymous_installation_id text not null,
  schema_version text not null,
  category text not null,
  module text not null,
  severity text not null default 'info',
  rosclaw_version text,
  robot_type text,
  skill_id text,
  task_id text,
  provider_type text,
  created_at timestamptz not null,
  received_at timestamptz not null default now(),
  privacy_level text not null default 'L0',
  redacted boolean not null default true,
  payload jsonb not null default '{}'::jsonb
);

create index if not exists idx_feedback_events_batch
  on public.feedback_events(batch_id);

create index if not exists idx_feedback_events_anon_id
  on public.feedback_events(anonymous_installation_id);

-- -------------------------------------------------
-- feedback_attachments
-- -------------------------------------------------
create table if not exists public.feedback_attachments (
  id uuid default gen_random_uuid() primary key,
  batch_id uuid references public.feedback_batches(id) on delete cascade,
  anonymous_installation_id text not null,
  storage_bucket text not null,
  storage_path text not null,
  file_name text not null,
  mime_type text not null,
  size_bytes bigint not null,
  sha256 text not null,
  redacted boolean not null default true,
  attachment_type text not null,
  created_at timestamptz not null default now()
);

create index if not exists idx_feedback_attachments_batch
  on public.feedback_attachments(batch_id);

-- -------------------------------------------------
-- feedback_delete_requests
-- -------------------------------------------------
create table if not exists public.feedback_delete_requests (
  id uuid default gen_random_uuid() primary key,
  anonymous_installation_id text not null,
  status text not null default 'pending',
  reason text,
  created_at timestamptz not null default now(),
  completed_at timestamptz
);

create index if not exists idx_feedback_delete_requests_anon_id
  on public.feedback_delete_requests(anonymous_installation_id);

-- -------------------------------------------------
-- Row Level Security
-- -------------------------------------------------
alter table public.telemetry_installations enable row level security;
alter table public.telemetry_events enable row level security;
alter table public.telemetry_daily_aggregates enable row level security;
alter table public.feedback_batches enable row level security;
alter table public.feedback_events enable row level security;
alter table public.feedback_attachments enable row level security;
alter table public.feedback_delete_requests enable row level security;

-- Service-role-only policies: direct client access is disabled.
-- Server-side API routes use the Supabase service role key.

create policy "service_role_only_telemetry_installations"
  on public.telemetry_installations
  for all
  to service_role
  using (true)
  with check (true);

create policy "service_role_only_telemetry_events"
  on public.telemetry_events
  for all
  to service_role
  using (true)
  with check (true);

create policy "service_role_only_telemetry_daily_aggregates"
  on public.telemetry_daily_aggregates
  for all
  to service_role
  using (true)
  with check (true);

create policy "service_role_only_feedback_batches"
  on public.feedback_batches
  for all
  to service_role
  using (true)
  with check (true);

create policy "service_role_only_feedback_events"
  on public.feedback_events
  for all
  to service_role
  using (true)
  with check (true);

create policy "service_role_only_feedback_attachments"
  on public.feedback_attachments
  for all
  to service_role
  using (true)
  with check (true);

create policy "service_role_only_feedback_delete_requests"
  on public.feedback_delete_requests
  for all
  to service_role
  using (true)
  with check (true);

-- Note: anon/authenticated roles have no access by design.
-- Admin dashboard reads through server-side API routes with service_role.
