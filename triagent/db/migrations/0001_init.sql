-- 0001_init: core domain tables for Triagent.
-- Mirrors triagent/models.py. Raw SQL, applied by triagent.db.migrate.

-- A GitHub issue. Primary key is the natural "repo#number" key so foreign keys
-- elsewhere stay single-column. Pull requests are filtered at ingest and never
-- stored, so there is no is_pull_request column.
CREATE TABLE issue (
    key         text PRIMARY KEY,                       -- "owner/repo#number"
    repo        text NOT NULL,                          -- "owner/name"
    number      integer NOT NULL,
    title       text NOT NULL,
    body        text,
    html_url    text NOT NULL,
    state       text NOT NULL CHECK (state IN ('open', 'closed')),
    labels      text[] NOT NULL DEFAULT '{}',
    language    text,
    created_at  timestamptz NOT NULL,                   -- opened on GitHub
    updated_at  timestamptz NOT NULL,                   -- last GitHub update
    source      text NOT NULL CHECK (source IN ('search', 'watchlist')),
    first_seen  timestamptz NOT NULL,                   -- first ingest sighting
    last_seen   timestamptz NOT NULL,                   -- latest ingest sighting
    raw         jsonb NOT NULL,                         -- full GitHub payload
    CONSTRAINT issue_repo_number_key UNIQUE (repo, number)
);

-- A scoring of an issue. Many rows may exist per issue across model/prompt
-- versions; downstream picks the most recent.
CREATE TABLE score (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    issue_key      text NOT NULL REFERENCES issue (key) ON DELETE CASCADE,
    solvability    double precision NOT NULL,
    skill_fit      double precision NOT NULL,
    difficulty     text NOT NULL CHECK (difficulty IN ('easy', 'medium', 'hard')),
    issue_type     text NOT NULL CHECK (issue_type IN ('bug', 'feature', 'docs', 'other')),
    model_version  text NOT NULL,
    prompt_version text NOT NULL,
    rationale      text NOT NULL,
    scored_at      timestamptz NOT NULL
);

-- Supports the "unscored issues" anti-join (issues with no matching score row)
-- and fast lookups of an issue's scores.
CREATE INDEX score_issue_key_idx ON score (issue_key);
-- Supports "issues by difficulty" filtering on the board.
CREATE INDEX score_difficulty_idx ON score (difficulty);

-- One pipeline execution (ingest / score / agent), with counters and room for
-- cost + latency observability.
CREATE TABLE run (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind        text NOT NULL CHECK (kind IN ('ingest', 'score', 'agent')),
    started_at  timestamptz NOT NULL,
    finished_at timestamptz,
    status      text NOT NULL CHECK (status IN ('running', 'success', 'error')),
    seen        integer NOT NULL DEFAULT 0,
    new         integer NOT NULL DEFAULT 0,
    updated     integer NOT NULL DEFAULT 0,
    cost_usd    numeric(10, 4),
    latency_ms  integer
);

-- Tracks a real PR opened to solve an issue, from open through merge.
CREATE TABLE solve_log (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    issue_key  text NOT NULL REFERENCES issue (key) ON DELETE CASCADE,
    pr_url     text NOT NULL,
    status     text NOT NULL CHECK (status IN ('open', 'review', 'merged', 'closed')),
    opened_at  timestamptz NOT NULL,
    merged_at  timestamptz,
    notes      text
);

CREATE INDEX solve_log_issue_key_idx ON solve_log (issue_key);
