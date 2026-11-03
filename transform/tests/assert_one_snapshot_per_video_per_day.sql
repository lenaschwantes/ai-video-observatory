select video_id, captured_on, count(*) as n
from {{ ref('stg_video_snapshots') }}
group by video_id, captured_on
having count(*) > 1
