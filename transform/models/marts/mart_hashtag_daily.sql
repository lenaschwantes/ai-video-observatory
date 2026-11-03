-- Hashtag-level daily series consumed by trend detection.

with exploded as (
    select captured_on, video_id, views_gained, unnest(hashtags) as tag
    from {{ ref('fct_video_daily') }}
)

select
    tag,
    captured_on,
    count(distinct video_id) as n_videos,
    sum(views_gained) as views_gained
from exploded
group by tag, captured_on
