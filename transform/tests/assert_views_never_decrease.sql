-- Cumulative counters should only go up. Platforms do occasionally remove
-- fake views, so this warns instead of failing the build.
{{ config(severity = 'warn') }}

select video_id, captured_on, views_gained
from {{ ref('fct_video_daily') }}
where views_gained < 0
