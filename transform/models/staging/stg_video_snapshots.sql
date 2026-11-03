-- One row per video per day. Reads only the normalized half of each bronze
-- envelope with an explicit schema (the contract); the raw payload stays in
-- bronze for reprocessing. If a day was ingested twice (e.g. once by a live
-- source and once by its fallback), the latest ingestion wins.

with landed as (
    select *
    from read_json(
        '{{ var("raw_path") }}/dt=*/*.jsonl',
        format = 'newline_delimited',
        hive_partitioning = true,
        columns = {
            source: 'varchar',
            captured_on: 'date',
            ingested_at: 'timestamptz',
            normalized: 'struct(
                video_id varchar, author varchar, description varchar,
                hashtags varchar[], music_id varchar, created_at timestamptz,
                views bigint, likes bigint, comments bigint, shares bigint
            )'
        }
    )
)

select
    normalized.video_id as video_id,
    captured_on,
    source,
    ingested_at,
    normalized.author as author,
    normalized.description as description,
    normalized.hashtags as hashtags,
    normalized.music_id as music_id,
    normalized.created_at as created_at,
    normalized.views as views,
    normalized.likes as likes,
    normalized.comments as comments,
    normalized.shares as shares
from landed
qualify row_number() over (partition by normalized.video_id, captured_on order by ingested_at desc) = 1
