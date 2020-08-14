""" Dependencies for java library. """

load("@rules_jvm_external//:defs.bzl", "maven_install")

def dp_java_deps():
    maven_install(
        artifacts = [
            # artifacts for building and testing
            "org.apache.commons:commons-math3:3.6.1",
            "com.google.auto.value:auto-value-annotations:1.7",
            "com.google.auto.value:auto-value:1.7",
            "com.google.code.findbugs:jsr305:3.0.2",
            "com.google.errorprone:error_prone_annotations:2.3.4",
            "com.google.guava:guava:28.2-jre",
            "com.google.protobuf:protobuf-java:3.11.4",
            # artifacts for testing only
            "org.mockito:mockito-core:3.3.0",
            "junit:junit:4.13",
            "com.google.truth:truth:1.0.1",
        ],
        repositories = [
            "https://jcenter.bintray.com/",
            "https://maven.google.com",
            "https://repo1.maven.org/maven2",
        ],
    )
