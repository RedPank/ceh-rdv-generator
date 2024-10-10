CREATE TABLE IF NOT EXISTS {{hub.hub_schema}}.{{hub.hub_name_only}} (
    {{hub.name}} int8 NOT NULL,
    {{hub.hub_field}} text NOT NULL,
    src_cd text NOT NULL,
    bk_type text NOT NULL,
    invalid_id int8 NOT NULL,
    version_id int8 NOT NULL
)
WITH (
  appendoptimized=true,
  orientation=column,
  compresslevel=1,
  compresstype=zstd
)
DISTRIBUTED BY ({{hub.name}});

DO $$

DECLARE
    v_res      numeric;
BEGIN

    SELECT COUNT(*)
      FROM information_schema.tables INTO v_res
     WHERE table_schema = '{{hub.hub_schema}}'
       AND table_name   = '{{hub.hub_name_only}}';

    IF v_res > 0 THEN
        EXECUTE 'SELECT COUNT(*) FROM {{hub.hub_schema}}.{{hub.hub_name_only}} WHERE {{ hub.name }} = -1'
            INTO v_res;

        IF v_res = 0 THEN
        EXECUTE 'INSERT INTO {{hub.hub_schema}}.{{hub.hub_name_only}} ({{hub.name}}, {{hub.hub_field}}, src_cd, bk_type, invalid_id, version_id)
                 VALUES (-1, ''~default~novalue~'', ''DEFAULT'', ''DEFAULT_NO_OBJECT'', 0, 0)';

            RAISE notice ' =  =  =  =  = -3.%. record has been added', clock_timestamp();
        ELSE
            RAISE notice ' =  =  =  =  = -3.%. record already exists, step skipped', clock_timestamp();
        END IF;

        EXECUTE 'SELECT COUNT(*) FROM {{hub.hub_schema}}.{{hub.hub_name_only}} WHERE {{hub.name}} = -2'
            INTO v_res;

        IF v_res = 0 THEN
        EXECUTE 'INSERT INTO {{hub.hub_schema}}.{{hub.hub_name_only}} ({{hub.name}}, {{hub.hub_field}}, src_cd, bk_type, invalid_id, version_id)
                 VALUES (-2, ''~default~unknownvalue~'', ''DEFAULT'', ''DEFAULT_UNKNOWN_OBJECT'', 0, 0)';
            RAISE notice ' =  =  =  =  = -3.%. record has been added', clock_timestamp();
        ELSE
            RAISE notice ' =  =  =  =  = -3.%. record already exists, step skipped', clock_timestamp();
        END IF;
    END IF;
END$$;