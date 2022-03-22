This image is based upon [`postgres:12`](https://hub.docker.com/_/postgres/) and adds PostgreSQL extension with several epsilon-DP aggregate functions. 

## How to build image

```
docker build --no-cache -t google/differential-privacy-postgres . 
```
and run
```
docker run -e POSTGRES_PASSWORD=password -p 5432:5432 google/differential-privacy-postgres
```

## Load extension

```
psql -U postgres -h localhost -p 5432
```
connect using password `password` and load the extension by calling
```
    CREATE EXTENSION anon_func;
```
## Run anon_func in Postgres

import data examples already present in image  with commands 

```
CREATE TABLE FruitEaten (
  uid integer,
  fruit character varying(20)
);
COPY fruiteaten(uid, fruit) FROM '/fruiteaten.csv' DELIMITER ',' CSV HEADER;
```

## Simple Count (Disclaimer This section is a part of C++ project)


In this table, each row represents one fruit eaten. So if person `1` eats two
`apple`s, then there will be two rows in the table with column values
`(1, apple)`. Consider a simple query counting how many of each fruit have been
eaten.

```
SELECT fruit, COUNT(fruit)
FROM FruitEaten
GROUP BY fruit;
```


Suppose that instead of getting the regular count, we want the differentially
private count with the privacy parameter ε=ln(3). The final product of the query
rewrite would be

```
SELECT result.fruit, result.number_eaten
FROM (
  SELECT per_person.fruit,
    ANON_SUM(per_person.fruit_count, LN(3)/2) as number_eaten,
    ANON_COUNT(uid, LN(3)/2) as number_eaters
    FROM(
      SELECT * , ROW_NUMBER() OVER (
        PARTITION BY uid
        ORDER BY random()
      ) as row_num
      FROM (
        SELECT fruit, uid, COUNT(fruit) as fruit_count
        FROM FruitEaten
        GROUP BY fruit, uid
      ) as per_person_raw
    ) as per_person
  WHERE per_person.row_num <= 5
  GROUP BY per_person.fruit
) as result
WHERE result.number_eaters > 50;

```

please for more infos or more accurate description please use [README](https://github.com/google/differential-privacy/blob/main/cc/README.md)

## Troubleshooting

If during startup stumble permission problem on file `docker-entrypoint-initdb.d` and `postgresql.conf.sample` please run on both files `chmod 755` command
