-- A collector that "succeeds" but returns far fewer rows than usual is the
-- most common silent failure. Flag any day below half of its trailing
-- 7-day average.

with daily as (
    select captured_on, count(*) as n
    from {{ ref('stg_video_snapshots') }}
    group by captured_on
),

scored as (
    select
        captured_on,
        n,
        avg(n) over (order by captured_on rows between 7 preceding and 1 preceding) as trailing_avg
    from daily
)

select *
from scored
where trailing_avg is not null and n < 0.5 * trailing_avg
