WITH tb_abt AS (
    
    SELECT t1.*,
        coalesce(t2.rankdriver,0 ) AS flChampion

    FROM "nekt_silver"."fs_f1_driver_all" AS t1

    LEFT JOIN "nekt_silver"."f1_champions" AS t2
    ON t1.driverid = t2.driverid
    AND year(t1.dt_ref) = t2.year

    WHERE t1.dt_ref >= date('2000-01-01')
    AND t1.dt_ref < date('2026-01-01')

    order by dt_ref desc, driverid
)

SELECT * FROM tb_abt