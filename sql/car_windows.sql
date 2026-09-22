DELETE FROM car;

INSERT INTO car
SELECT
    a.event_id,
    w.window,
    SUM(a.ar) AS car,
    SUM(a.sar) / SQRT(COUNT(*)) AS scar,
    COUNT(*) AS n_days
FROM abnormal_returns AS a
JOIN windows AS w
  ON a.rel_day BETWEEN w.start_day AND w.end_day
GROUP BY a.event_id, w.window
HAVING COUNT(*) = (MAX(w.end_day) - MIN(w.start_day) + 1);
