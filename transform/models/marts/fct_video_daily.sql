-- Daily velocity per video. views_gained compares with the previous snapshot
-- of the same video; on its first sighting the video's whole view count counts
-- as gained, since it entered the trending feed that day.

select
    video_id,
    captured_on,
    author,
    hashtags,
    music_id,
    created_at,
    views,
    likes,
    comments,
    shares,
    views - lag(views) over w is null as is_first_sighting,
    coalesce(views - lag(views) over w, views) as views_gained,
    likes::double / nullif(views, 0) as like_rate
from {{ ref('stg_video_snapshots') }}
window w as (partition by video_id order by captured_on)
