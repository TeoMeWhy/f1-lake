WITH
	-- 1. Determina todas as datas de referência únicas (corridas de 2024 a 2025)
	dim_dates AS (
		SELECT DISTINCT date(date) AS dt_ref,
					 year AS ref_year
		FROM f1_results
		WHERE year >= 2024 AND year <= 2025 -- isso é parametrizável
	),

	-- 2. Cruzamento: Para cada data de referência, encontre todas as sessões que ocorreram antes ou na data de referência
	past_sessions AS (
		SELECT 
			d.dt_ref, 
			d.ref_year,
			f.*
		FROM dim_dates d
		INNER JOIN f1_results as f
			ON date(f.date) <= d.dt_ref
	),

	-- 5. Filtra pilotos elegíveis, considerando apenas aqueles que participaram de sessões nos últimos 2 anos antes de cada data de referência
	eligible_drivers AS (
		SELECT DISTINCT p.dt_ref, p.driverid
		FROM past_sessions p
		WHERE ref_year - year <= 2
		order by dt_ref desc
	),

	-- 6. Obtém as rodadas distintas que ocorreram antes de cada data de referência
	distinct_rounds AS (
		SELECT DISTINCT dt_ref, year, roundnumber
		FROM past_sessions
	),

	-- 7. Ordena a lista de rodadas para cada data de referência e atribui um número de linha
	ranked_rounds AS (
		SELECT 
			dt_ref, 
			year, 
			roundnumber,
			ROW_NUMBER() OVER (PARTITION BY dt_ref ORDER BY year DESC, roundnumber DESC) AS rn
		FROM distinct_rounds
	),

	-- 8. Mantém apenas as últimas 20 rodadas para cada data de referência
	last_rounds AS (
		SELECT dt_ref, year, roundnumber
		FROM ranked_rounds
		WHERE rn <= 20  -- isso é parametrizável

	),

	-- 9. Junta tabela de resultados com os pilotos elegíveis em cada uma das datas de referência
	tb_results AS (
		SELECT p.*
		FROM past_sessions p
		INNER JOIN eligible_drivers e
			ON p.dt_ref = e.dt_ref 
			AND p.driverid = e.driverid
		INNER JOIN last_rounds r
			ON p.dt_ref = r.dt_ref 
			AND p.year = r.year 
			AND p.roundnumber = r.roundnumber
	),

	-- 10. Estatística agrupada por data de referência e piloto
	tb_stats AS (
  SELECT
    dt_ref,
    driverid,
    count(DISTINCT YEAR) AS qtd_seasons,
    count(*) AS qtd_sessions,
    sum( CASE WHEN ( status = 'Finished' OR status LIKE '+%') THEN 1 ELSE 0 END ) AS qtde_sessions_finished,
    sum( CASE WHEN mode = 'Race' THEN 1 ELSE 0 END ) AS qtd_race,
    sum( CASE WHEN mode = 'Race' AND (status = 'Finished' OR status LIKE '+%') THEN 1 ELSE 0 END ) AS qtde_sessions_finished_race,
    sum( CASE WHEN mode = 'Sprint' THEN 1 ELSE 0 END ) AS qtd_sprint,
    sum( CASE WHEN mode = 'Sprint' AND (status = 'Finished' OR status LIKE '+%') THEN 1 ELSE 0 END ) AS qtde_sessions_finished_sprint,
    sum( CASE WHEN POSITION = 1 THEN 1 ELSE 0 END ) AS qtde_1Pos,
    sum( CASE WHEN POSITION = 1 AND MODE = 'Race' THEN 1 ELSE 0 END ) AS qtde_1Pos_race,
    sum( CASE WHEN POSITION = 1 AND MODE = 'Sprint' THEN 1 ELSE 0 END ) AS qtde_1Pos_sprint,
    sum( CASE WHEN POSITION <= 3 THEN 1 ELSE 0 END ) AS qtde_podios,
    sum( CASE WHEN POSITION <= 3 AND mode = 'Race' THEN 1 ELSE 0 END ) AS qtde_podios_race,
    sum( CASE WHEN POSITION <= 3 AND mode = 'Sprint' THEN 1 ELSE 0 END ) AS qtde_podios_sprint,
    sum( CASE WHEN POSITION <= 5 THEN 1 ELSE 0 END ) AS qtde_pos5,
    sum( CASE WHEN POSITION <= 5 AND mode = 'Race' THEN 1 ELSE 0 END ) AS qtde_pos5_race,
    sum( CASE WHEN POSITION <= 5 AND mode = 'Sprint' THEN 1 ELSE 0 END ) AS qtde_pos5_sprint,
    sum( CASE WHEN gridposition <= 5 THEN 1 ELSE 0 END ) AS qtde_gridpos5,
    sum( CASE WHEN gridposition <= 5 AND mode = 'Race' THEN 1 ELSE 0 END ) AS qtde_gridpos5_race,
    sum( CASE WHEN gridposition <= 5 AND mode = 'Sprint' THEN 1 ELSE 0 END ) AS qtde_gridpos5_sprint,
    sum(points) AS qtde_points,
    sum( CASE WHEN mode = 'Race' THEN points END ) AS qtde_points_race,
    sum( CASE WHEN mode = 'Sprint' THEN points END ) AS qtde_points_sprint,
    avg(gridposition) AS avg_gridposition,
    avg( CASE WHEN mode = 'Race' THEN gridposition END ) AS avg_gridposition_race,
    avg( CASE WHEN mode = 'Sprint' THEN gridposition END ) AS avg_gridposition_sprint,
    avg(POSITION) AS avg_position,
    avg( CASE WHEN mode = 'Race' THEN POSITION END ) AS avg_position_race,
    avg( CASE WHEN mode = 'Sprint' THEN POSITION END ) AS avg_position_sprint,
    sum( CASE WHEN gridposition = 1 THEN 1 ELSE 0 END ) AS qtde_1_gridposition,
    sum( CASE WHEN gridposition = 1 AND mode = 'Race' THEN 1 ELSE 0 END ) AS qtde_1_gridposition_race,
    sum( CASE WHEN gridposition = 1 AND mode = 'Sprint' THEN 1 ELSE 0 END ) AS qtde_1_gridposition_sprint,
    sum( CASE WHEN gridposition = 1 AND POSITION = 1 THEN 1 ELSE 0 END ) AS qtde_pole_win,
    sum( CASE WHEN gridposition = 1 AND POSITION = 1 AND mode = 'Race' THEN 1 ELSE 0 END ) AS qtde_pole_win_race,
    sum( CASE WHEN gridposition = 1 AND POSITION = 1 AND mode = 'Sprint' THEN 1 ELSE 0 END ) AS qtde_pole_win_sprint,
    sum( CASE WHEN points > 0 THEN 1 ELSE 0 END ) AS qtd_sessions_with_points,
    sum( CASE WHEN mode = 'Race' AND points > 0 THEN 1 ELSE 0 END ) AS qtd_sessions_with_points_race,
    sum( CASE WHEN mode = 'Sprint' AND points > 0 THEN 1 ELSE 0 END ) AS qtd_sessions_with_points_sprint,
    sum( CASE WHEN POSITION < gridPOSITION THEN 1 ELSE 0 END ) AS qtde_sessions_with_overtake,
    sum( CASE WHEN mode = 'Race' AND POSITION < gridPOSITION THEN 1 ELSE 0 END ) AS qtde_sessions_with_overtake_race,
    sum( CASE WHEN mode = 'Sprint' AND POSITION < gridPOSITION THEN 1 ELSE 0 END ) AS qtde_sessions_with_overtake_sprint,
    avg(gridPOSITION - POSITION) AS avg_overtake,
    avg( CASE WHEN mode = 'Race' THEN gridPOSITION - POSITION END ) AS avg_overtake_race,
    avg( CASE WHEN mode = 'Sprint' THEN gridPOSITION - POSITION END ) AS avg_overtake_sprint
FROM tb_results
GROUP BY dt_ref, driverid

)

SELECT *
FROM tb_stats
ORDER BY dt_ref desc, driverid
