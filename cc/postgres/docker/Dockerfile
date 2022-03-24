FROM postgres:12 AS build

ENV PROTOC_VERSION=1.3


# Install the packages which will be required to get everything to compile
RUN apt-get update \
    && apt-get install -f -y --no-install-recommends \
        software-properties-common \
        build-essential \
        pkg-config \
        git \
        curl \
        libreadline-dev \
        bison \
        flex \
        postgresql-server-dev-$PG_MAJOR \
    && add-apt-repository "deb http://ftp.debian.org/debian testing main contrib" \
    && apt-get update && apt-get install -f -y --no-install-recommends \
        libprotobuf-c-dev=$PROTOC_VERSION.* \
    && rm -rf /var/lib/apt/lists/*

#Install Bazel to build code
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list \
  && curl https://bazel.build/bazel-release.pub.gpg | apt-key add -


RUN apt-get update \
  && apt-get install -y bazel=4.1.0 \
  && rm -rf /var/lib/apt/lists/*


##Download differential privacy module


RUN git clone https://github.com/google/differential-privacy.git --single-branch /tmp/differential-privacy \
    && cd /tmp/differential-privacy \
    && git checkout fc4f2abda5052f654539fc128 \
    && cd /tmp/differential-privacy/cc \
    && bazel build postgres/anon_func.so




FROM postgres:12

RUN apt-get update \
    && apt-get install -f -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository "deb http://ftp.debian.org/debian testing main contrib" \
    && apt-get update && apt-get install -f -y --no-install-recommends \
        libprotobuf-c1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /tmp/differential-privacy/cc/bazel-bin/postgres/anon_func.so  /usr/lib/postgresql/$PG_MAJOR/lib/
COPY --from=build /tmp/differential-privacy/cc/postgres/anon_func.control /usr/share/postgresql/$PG_MAJOR/extension/
COPY --from=build /tmp/differential-privacy/cc/postgres/anon_func--1.0.0.sql /usr/share/postgresql/$PG_MAJOR/extension/

#Copy Dataset sample
COPY --from=build /tmp/differential-privacy/cc/postgres/fruiteaten.csv  /
COPY --from=build /tmp/differential-privacy/cc/postgres/shirts.csv  /

# Copy the custom configuration which will be passed down to the server (using a .sample file is the preferred way of doing it by
# the base Docker image)
COPY postgresql.conf.sample /usr/share/postgresql/postgresql.conf.sample
RUN chmod 755 /usr/share/postgresql/postgresql.conf.sample

# Copy the script which will initialize the replication permissions
COPY /docker-entrypoint-initdb.d /docker-entrypoint-initdb.d
RUN chmod 755 /docker-entrypoint-initdb.d
