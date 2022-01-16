This image is based upon [`postgres:12`](https://hub.docker.com/_/postgres/) and adds PostgreSQL extension with several epsilon-DP aggregate functions. 

# How to build this image

```
docker build --no-cache -t google/differential-privacy-postgres . 
```
and run
```
docker run google/differential-privacy-postgres -e POSTGRES_PASSWORD=password
```

* In PostgreSQL, load the extension by calling

    ```
    CREATE EXTENSION anon_func;
    ```

# How to use this image

This image is used in the same manner as the [`postgres:12.0`](https://hub.docker.com/_/postgres/) image, though the `/usr/share/postgresql/postgresql.conf.sample` file configures the logical decoding feature:

```
# LOGGING
log_min_error_statement = fatal

# CONNECTION
listen_addresses = '*'

# MODULES
shared_preload_libraries = 'anon_func'
```
Data provied by `fruiteaten.csv` and `shirts.csv` can be found `/` directory
then to try example typing :
```
CREATE TABLE FruitEaten (
  uid integer,
  fruit character varying(20)
);
COPY fruiteaten(uid, fruit) FROM '/fruiteaten.csv' DELIMITER ',' CSV HEADER;
```