#
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

""" Declares dependencies of Privacy on Beam """

load("@bazel_gazelle//:deps.bzl", "go_repository")

def privacy_on_beam_deps():
    """ Macro to include Privacy on Beam's critical dependencies in a WORKSPACE.

        Generated automatically by gazelle.

    """
    go_repository(
        name = "cat_dario_mergo",
        importpath = "dario.cat/mergo",
        sum = "h1:AGCNq9Evsj31mOgNPcLyXc+4PNABt905YmuqPYYpBWk=",
        version = "v1.0.0",
    )
    go_repository(
        name = "co_honnef_go_tools",
        importpath = "honnef.co/go/tools",
        sum = "h1:qTakTkI6ni6LFD5sBwwsdSO+AQqbSIxOauHTTQKZ/7o=",
        version = "v0.1.3",
    )
    go_repository(
        name = "com_github_ajstarks_deck",
        importpath = "github.com/ajstarks/deck",
        sum = "h1:7kQgkwGRoLzC9K0oyXdJo7nve/bynv/KwUsxbiTlzAM=",
        version = "v0.0.0-20200831202436-30c9fc6549a9",
    )
    go_repository(
        name = "com_github_ajstarks_deck_generate",
        importpath = "github.com/ajstarks/deck/generate",
        sum = "h1:iXUgAaqDcIUGbRoy2TdeofRG/j1zpGRSEmNK05T+bi8=",
        version = "v0.0.0-20210309230005-c3f852c02e19",
    )
    go_repository(
        name = "com_github_ajstarks_svgo",
        importpath = "github.com/ajstarks/svgo",
        sum = "h1:slYM766cy2nI3BwyRiyQj/Ud48djTMtMebDqepE95rw=",
        version = "v0.0.0-20211024235047-1546f124cd8b",
    )
    go_repository(
        name = "com_github_andybalholm_stroke",
        importpath = "github.com/andybalholm/stroke",
        sum = "h1:uF5Q/hWnDU1XZeT6CsrRSxHLroUSEYYO3kgES+yd+So=",
        version = "v0.0.0-20221221101821-bd29b49d73f0",
    )
    go_repository(
        name = "com_github_apache_arrow_go_arrow",
        importpath = "github.com/apache/arrow/go/arrow",
        sum = "h1:byKBBF2CKWBjjA4J1ZL2JXttJULvWSl50LegTyRZ728=",
        version = "v0.0.0-20200730104253-651201b0f516",
    )
    go_repository(
        name = "com_github_apache_arrow_go_v14",
        importpath = "github.com/apache/arrow/go/v14",
        sum = "h1:N8OkaJEOfI3mEZt07BIkvo4sC6XDbL+48MBPWO5IONw=",
        version = "v14.0.2",
    )
    go_repository(
        name = "com_github_apache_beam_sdks_v2",
        build_file_proto_mode = "disable_global",  # See https://github.com/bazelbuild/rules_go/issues/2186#issuecomment-523028281
        importpath = "github.com/apache/beam/sdks/v2",
        sum = "h1:zVoIJc4/TEpxBAakzPSNVEJ3gdafMFFUByb6BVUaPHs=",
        version = "v2.55.0",
    )
    go_repository(
        name = "com_github_apache_thrift",
        importpath = "github.com/apache/thrift",
        sum = "h1:cMd2aj52n+8VoAtvSvLn4kDC3aZ6IAkBuqWQ2IDu7wo=",
        version = "v0.17.0",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go",
        importpath = "github.com/aws/aws-sdk-go",
        sum = "h1:brux2dRrlwCF5JhTL7MUT3WUwo9zfDHZZp3+g3Mvlmo=",
        version = "v1.34.0",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2",
        importpath = "github.com/aws/aws-sdk-go-v2",
        sum = "h1:/uiG1avJRgLGiQM9X3qJM8+Qa6KRGK5rRPuXE0HUM+w=",
        version = "v1.25.2",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_aws_protocol_eventstream",
        importpath = "github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream",
        sum = "h1:ZY3108YtBNq96jNZTICHxN1gSBSbnvIdYwwqnvCV4Mc=",
        version = "v1.5.1",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_config",
        importpath = "github.com/aws/aws-sdk-go-v2/config",
        sum = "h1:AhfWb5ZwimdsYTgP7Od8E9L1u4sKmDW2ZVeLcf2O42M=",
        version = "v1.27.4",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_credentials",
        importpath = "github.com/aws/aws-sdk-go-v2/credentials",
        sum = "h1:h5Vztbd8qLppiPwX+y0Q6WiwMZgpd9keKe2EAENgAuI=",
        version = "v1.17.4",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_feature_ec2_imds",
        importpath = "github.com/aws/aws-sdk-go-v2/feature/ec2/imds",
        sum = "h1:AK0J8iYBFeUk2Ax7O8YpLtFsfhdOByh2QIkHmigpRYk=",
        version = "v1.15.2",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_feature_s3_manager",
        importpath = "github.com/aws/aws-sdk-go-v2/feature/s3/manager",
        sum = "h1:wuOjvalpd2CnXffks74Vq6n3yv9vunKCoy4R1sjStGk=",
        version = "v1.13.8",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_internal_configsources",
        importpath = "github.com/aws/aws-sdk-go-v2/internal/configsources",
        sum = "h1:bNo4LagzUKbjdxE0tIcR9pMzLR2U/Tgie1Hq1HQ3iH8=",
        version = "v1.3.2",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_internal_endpoints_v2",
        importpath = "github.com/aws/aws-sdk-go-v2/internal/endpoints/v2",
        sum = "h1:EtOU5jsPdIQNP+6Q2C5e3d65NKT1PeCiQk+9OdzO12Q=",
        version = "v2.6.2",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_internal_ini",
        importpath = "github.com/aws/aws-sdk-go-v2/internal/ini",
        sum = "h1:hT8rVHwugYE2lEfdFE0QWVo81lF7jMrYJVDWI+f+VxU=",
        version = "v1.8.0",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_internal_v4a",
        importpath = "github.com/aws/aws-sdk-go-v2/internal/v4a",
        sum = "h1:lMwCXiWJlrtZot0NJTjbC8G9zl+V3i68gBTBBvDeEXA=",
        version = "v1.2.3",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_service_internal_accept_encoding",
        importpath = "github.com/aws/aws-sdk-go-v2/service/internal/accept-encoding",
        sum = "h1:EyBZibRTVAs6ECHZOw5/wlylS9OcTzwyjeQMudmREjE=",
        version = "v1.11.1",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_service_internal_checksum",
        importpath = "github.com/aws/aws-sdk-go-v2/service/internal/checksum",
        sum = "h1:xbwRyCy7kXrOj89iIKLB6NfE2WCpP9HoKyk8dMDvnIQ=",
        version = "v1.2.3",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_service_internal_presigned_url",
        importpath = "github.com/aws/aws-sdk-go-v2/service/internal/presigned-url",
        sum = "h1:5ffmXjPtwRExp1zc7gENLgCPyHFbhEPwVTkTiH9niSk=",
        version = "v1.11.2",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_service_internal_s3shared",
        importpath = "github.com/aws/aws-sdk-go-v2/service/internal/s3shared",
        sum = "h1:KV0z2RDc7euMtg8aUT1czv5p29zcLlXALNFsd3jkkEc=",
        version = "v1.16.3",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_service_s3",
        importpath = "github.com/aws/aws-sdk-go-v2/service/s3",
        sum = "h1:NnduxUd9+Fq9DcCDdJK8v6l9lR1xDX4usvog+JuQAno=",
        version = "v1.42.2",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_service_sso",
        importpath = "github.com/aws/aws-sdk-go-v2/service/sso",
        sum = "h1:utEGkfdQ4L6YW/ietH7111ZYglLJvS+sLriHJ1NBJEQ=",
        version = "v1.20.1",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_service_ssooidc",
        importpath = "github.com/aws/aws-sdk-go-v2/service/ssooidc",
        sum = "h1:9/GylMS45hGGFCcMrUZDVayQE1jYSIN6da9jo7RAYIw=",
        version = "v1.23.1",
    )
    go_repository(
        name = "com_github_aws_aws_sdk_go_v2_service_sts",
        importpath = "github.com/aws/aws-sdk-go-v2/service/sts",
        sum = "h1:3I2cBEYgKhrWlwyZgfpSO2BpaMY1LHPqXYk/QGlu2ew=",
        version = "v1.28.1",
    )
    go_repository(
        name = "com_github_aws_smithy_go",
        importpath = "github.com/aws/smithy-go",
        sum = "h1:4SZlSlMr36UEqC7XOyRVb27XMeZubNcBNN+9IgEPIQw=",
        version = "v1.20.1",
    )
    go_repository(
        name = "com_github_azure_go_ansiterm",
        importpath = "github.com/Azure/go-ansiterm",
        sum = "h1:UQHMgLO+TxOElx5B5HZ4hJQsoJ/PvUvKRhJHDQXO8P8=",
        version = "v0.0.0-20210617225240-d185dfc1b5a1",
    )
    go_repository(
        name = "com_github_boombuler_barcode",
        importpath = "github.com/boombuler/barcode",
        sum = "h1:NDBbPmhS+EqABEs5Kg3n/5ZNjy73Pz7SIV+KCeqyXcs=",
        version = "v1.0.1",
    )
    go_repository(
        name = "com_github_burntsushi_toml",
        importpath = "github.com/BurntSushi/toml",
        sum = "h1:WXkYYl6Yr3qBf1K79EBnL4mak0OimBfB0XUf9Vl28OQ=",
        version = "v0.3.1",
    )
    go_repository(
        name = "com_github_campoy_embedmd",
        importpath = "github.com/campoy/embedmd",
        sum = "h1:V4kI2qTJJLf4J29RzI/MAt2c3Bl4dQSYPuflzwFH2hY=",
        version = "v1.0.0",
    )
    go_repository(
        name = "com_github_cenkalti_backoff_v4",
        importpath = "github.com/cenkalti/backoff/v4",
        sum = "h1:y4OZtCnogmCPw98Zjyt5a6+QwPLGkiQsYW5oUqylYbM=",
        version = "v4.2.1",
    )
    go_repository(
        name = "com_github_census_instrumentation_opencensus_proto",
        importpath = "github.com/census-instrumentation/opencensus-proto",
        sum = "h1:iKLQ0xPNFxR/2hzXZMrBo8f1j86j5WHzznCCQxV/b8g=",
        version = "v0.4.1",
    )
    go_repository(
        name = "com_github_cespare_xxhash_v2",
        importpath = "github.com/cespare/xxhash/v2",
        sum = "h1:DC2CZ1Ep5Y4k3ZQ899DldepgrayRUGE6BBZ/cd9Cj44=",
        version = "v2.2.0",
    )
    go_repository(
        name = "com_github_chromedp_cdproto",
        importpath = "github.com/chromedp/cdproto",
        sum = "h1:aPflPkRFkVwbW6dmcVqfgwp1i+UWGFH6VgR1Jim5Ygc=",
        version = "v0.0.0-20230802225258-3cf4e6d46a89",
    )
    go_repository(
        name = "com_github_chromedp_chromedp",
        importpath = "github.com/chromedp/chromedp",
        sum = "h1:dKtNz4kApb06KuSXoTQIyUC2TrA0fhGDwNZf3bcgfKw=",
        version = "v0.9.2",
    )
    go_repository(
        name = "com_github_chromedp_sysutil",
        importpath = "github.com/chromedp/sysutil",
        sum = "h1:+ZxhTpfpZlmchB58ih/LBHX52ky7w2VhQVKQMucy3Ic=",
        version = "v1.0.0",
    )
    go_repository(
        name = "com_github_chzyer_readline",
        importpath = "github.com/chzyer/readline",
        sum = "h1:upd/6fQk4src78LMRzh5vItIt361/o4uq553V8B5sGI=",
        version = "v1.5.1",
    )
    go_repository(
        name = "com_github_client9_misspell",
        importpath = "github.com/client9/misspell",
        sum = "h1:ta993UF76GwbvJcIo3Y68y/M3WxlpEHPWIGDkJYwzJI=",
        version = "v0.3.4",
    )
    go_repository(
        name = "com_github_cncf_udpa_go",
        importpath = "github.com/cncf/udpa/go",
        sum = "h1:QQ3GSy+MqSHxm/d8nCtnAiZdYFd45cYZPs8vOOIYKfk=",
        version = "v0.0.0-20220112060539-c52dc94e7fbe",
    )
    go_repository(
        name = "com_github_cncf_xds_go",
        importpath = "github.com/cncf/xds/go",
        sum = "h1:jQCWAUqqlij9Pgj2i/PB79y4KOPYVyFYdROxgaCwdTQ=",
        version = "v0.0.0-20231128003011-0fa0005c9caa",
    )
    go_repository(
        name = "com_github_containerd_containerd",
        importpath = "github.com/containerd/containerd",
        sum = "h1:lfGKw3eU35sjV0aG2eYZTiwFEY1pCzxdzicHP3SZILw=",
        version = "v1.7.11",
    )
    go_repository(
        name = "com_github_containerd_log",
        importpath = "github.com/containerd/log",
        sum = "h1:TCJt7ioM2cr/tfR8GPbGf9/VRAX8D2B4PjzCpfX540I=",
        version = "v0.1.0",
    )
    go_repository(
        name = "com_github_cpuguy83_dockercfg",
        importpath = "github.com/cpuguy83/dockercfg",
        sum = "h1:/FpZ+JaygUR/lZP2NlFI2DVfrOEMAIKP5wWEJdoYe9E=",
        version = "v0.3.1",
    )
    go_repository(
        name = "com_github_davecgh_go_spew",
        importpath = "github.com/davecgh/go-spew",
        sum = "h1:vj9j/u1bqnvCEfJOwUhtlOARqs3+rkHYY13jYWTU97c=",
        version = "v1.1.1",
    )
    go_repository(
        name = "com_github_distribution_reference",
        importpath = "github.com/distribution/reference",
        sum = "h1:/FUIFXtfc/x2gpa5/VGfiGLuOIdYa1t65IKK2OFGvA0=",
        version = "v0.5.0",
    )
    go_repository(
        name = "com_github_docker_docker",
        importpath = "github.com/docker/docker",
        sum = "h1:UmQydMduGkrD5nQde1mecF/YnSbTOaPeFIeP5C4W+DE=",
        version = "v25.0.5+incompatible",
    )
    go_repository(
        name = "com_github_docker_go_connections",
        importpath = "github.com/docker/go-connections",
        sum = "h1:USnMq7hx7gwdVZq1L49hLXaFtUdTADjXGp+uj1Br63c=",
        version = "v0.5.0",
    )
    go_repository(
        name = "com_github_docker_go_units",
        importpath = "github.com/docker/go-units",
        sum = "h1:69rxXcBk27SvSaaxTtLh/8llcHD8vYHT7WSdRZ/jvr4=",
        version = "v0.5.0",
    )
    go_repository(
        name = "com_github_dustin_go_humanize",
        importpath = "github.com/dustin/go-humanize",
        sum = "h1:GzkhY7T5VNhEkwH0PVJgjz+fX1rhBrR7pRT3mDkpeCY=",
        version = "v1.0.1",
    )
    go_repository(
        name = "com_github_envoyproxy_go_control_plane",
        importpath = "github.com/envoyproxy/go-control-plane",
        sum = "h1:4X+VP1GHd1Mhj6IB5mMeGbLCleqxjletLK6K0rbxyZI=",
        version = "v0.12.0",
    )
    go_repository(
        name = "com_github_envoyproxy_protoc_gen_validate",
        importpath = "github.com/envoyproxy/protoc-gen-validate",
        sum = "h1:gVPz/FMfvh57HdSJQyvBtF00j8JU4zdyUgIUNhlgg0A=",
        version = "v1.0.4",
    )
    go_repository(
        name = "com_github_felixge_httpsnoop",
        importpath = "github.com/felixge/httpsnoop",
        sum = "h1:NFTV2Zj1bL4mc9sqWACXbQFVBBg2W3GPvqp8/ESS2Wg=",
        version = "v1.0.4",
    )
    go_repository(
        name = "com_github_fogleman_gg",
        importpath = "github.com/fogleman/gg",
        sum = "h1:/7zJX8F6AaYQc57WQCyN9cAIz+4bCJGO9B+dyW29am8=",
        version = "v1.3.0",
    )
    go_repository(
        name = "com_github_frankban_quicktest",
        importpath = "github.com/frankban/quicktest",
        sum = "h1:+cqqvzZV87b4adx/5ayVOaYZ2CrvM4ejQvUdBzPPUss=",
        version = "v1.14.0",
    )
    go_repository(
        name = "com_github_fsouza_fake_gcs_server",
        importpath = "github.com/fsouza/fake-gcs-server",
        sum = "h1:56/U4rKY081TaNbq0gHWi7/71UxC2KROqcnrD9BRJhs=",
        version = "v1.47.7",
    )
    go_repository(
        name = "com_github_go_fonts_dejavu",
        importpath = "github.com/go-fonts/dejavu",
        sum = "h1:3XlHi0JBYX+Cp8n98c6qSoHrxPa4AUKDMKdrh/0sUdk=",
        version = "v0.3.2",
    )
    go_repository(
        name = "com_github_go_fonts_latin_modern",
        importpath = "github.com/go-fonts/latin-modern",
        sum = "h1:M+Sq24Dp0ZRPf3TctPnG1MZxRblqyWC/cRUL9WmdaFc=",
        version = "v0.3.2",
    )
    go_repository(
        name = "com_github_go_fonts_liberation",
        importpath = "github.com/go-fonts/liberation",
        sum = "h1:XuwG0vGHFBPRRI8Qwbi5tIvR3cku9LUfZGq/Ar16wlQ=",
        version = "v0.3.2",
    )
    go_repository(
        name = "com_github_go_fonts_stix",
        importpath = "github.com/go-fonts/stix",
        sum = "h1:v9krocr13J1llaOHLEol1eaHsv8S43UuFX/1bFgEJJ4=",
        version = "v0.2.2",
    )
    go_repository(
        name = "com_github_go_latex_latex",
        importpath = "github.com/go-latex/latex",
        sum = "h1:DfZQkvEbdmOe+JK2TMtBM+0I9GSdzE2y/L1/AmD8xKc=",
        version = "v0.0.0-20231108140139-5c1ce85aa4ea",
    )
    go_repository(
        name = "com_github_go_logr_logr",
        importpath = "github.com/go-logr/logr",
        sum = "h1:pKouT5E8xu9zeFC39JXRDukb6JFQPXM5p5I91188VAQ=",
        version = "v1.4.1",
    )
    go_repository(
        name = "com_github_go_logr_stdr",
        importpath = "github.com/go-logr/stdr",
        sum = "h1:hSWxHoqTgW2S2qGc0LTAI563KZ5YKYRhT3MFKZMbjag=",
        version = "v1.2.2",
    )
    go_repository(
        name = "com_github_go_ole_go_ole",
        importpath = "github.com/go-ole/go-ole",
        sum = "h1:/Fpf6oFPoeFik9ty7siob0G6Ke8QvQEuVcuChpwXzpY=",
        version = "v1.2.6",
    )
    go_repository(
        name = "com_github_go_pdf_fpdf",
        importpath = "github.com/go-pdf/fpdf",
        sum = "h1:PPvSaUuo1iMi9KkaAn90NuKi+P4gwMedWPHhj8YlJQw=",
        version = "v0.9.0",
    )
    go_repository(
        name = "com_github_go_sql_driver_mysql",
        importpath = "github.com/go-sql-driver/mysql",
        sum = "h1:lUIinVbN1DY0xBg0eMOzmmtGoHwWBbvnWubQUrtU8EI=",
        version = "v1.7.1",
    )
    go_repository(
        name = "com_github_go_text_typesetting",
        importpath = "github.com/go-text/typesetting",
        sum = "h1:FQivqchis6bE2/9uF70M2gmmLpe82esEm2QadL0TEJo=",
        version = "v0.0.0-20230803102845-24e03d8b5372",
    )
    go_repository(
        name = "com_github_gobwas_httphead",
        importpath = "github.com/gobwas/httphead",
        sum = "h1:exrUm0f4YX0L7EBwZHuCF4GDp8aJfVeBrlLQrs6NqWU=",
        version = "v0.1.0",
    )
    go_repository(
        name = "com_github_gobwas_pool",
        importpath = "github.com/gobwas/pool",
        sum = "h1:xfeeEhW7pwmX8nuLVlqbzVc7udMDrwetjEv+TZIz1og=",
        version = "v0.2.1",
    )
    go_repository(
        name = "com_github_gobwas_ws",
        importpath = "github.com/gobwas/ws",
        sum = "h1:F2aeBZrm2NDsc7vbovKrWSogd4wvfAxg0FQ89/iqOTk=",
        version = "v1.2.1",
    )
    go_repository(
        name = "com_github_goccmack_gocc",
        importpath = "github.com/goccmack/gocc",
        sum = "h1:FSii2UQeSLngl3jFoR4tUKZLprO7qUlh/TKKticc0BM=",
        version = "v0.0.0-20230228185258-2292f9e40198",
    )
    go_repository(
        name = "com_github_goccy_go_json",
        importpath = "github.com/goccy/go-json",
        sum = "h1:CrxCmQqYDkv1z7lO7Wbh2HN93uovUHgrECaO5ZrCXAU=",
        version = "v0.10.2",
    )
    go_repository(
        name = "com_github_gogo_protobuf",
        importpath = "github.com/gogo/protobuf",
        sum = "h1:Ov1cvc58UF3b5XjBnZv7+opcTcQFZebYjWzi34vdm4Q=",
        version = "v1.3.2",
    )
    go_repository(
        name = "com_github_golang_freetype",
        importpath = "github.com/golang/freetype",
        sum = "h1:DACJavvAHhabrF08vX0COfcOBJRhZ8lUbR+ZWIs0Y5g=",
        version = "v0.0.0-20170609003504-e2365dfdc4a0",
    )
    go_repository(
        name = "com_github_golang_glog",
        importpath = "github.com/golang/glog",
        sum = "h1:uCdmnmatrKCgMBlM4rMuJZWOkPDqdbZPnrMXDY4gI68=",
        version = "v1.2.0",
    )
    go_repository(
        name = "com_github_golang_groupcache",
        importpath = "github.com/golang/groupcache",
        sum = "h1:oI5xCqsCo564l8iNU+DwB5epxmsaqB+rhGL0m5jtYqE=",
        version = "v0.0.0-20210331224755-41bb18bfe9da",
    )
    go_repository(
        name = "com_github_golang_mock",
        importpath = "github.com/golang/mock",
        sum = "h1:ErTB+efbowRARo13NNdxyJji2egdxLGQhRaY+DUumQc=",
        version = "v1.6.0",
    )
    go_repository(
        name = "com_github_golang_protobuf",
        importpath = "github.com/golang/protobuf",
        sum = "h1:i7eJL8qZTpSEXOPTxNKhASYpMn+8e5Q6AdndVa1dWek=",
        version = "v1.5.4",
    )
    go_repository(
        name = "com_github_golang_snappy",
        importpath = "github.com/golang/snappy",
        sum = "h1:yAGX7huGHXlcLOEtBnF4w7FQwA26wojNCwOYAEhLjQM=",
        version = "v0.0.4",
    )
    go_repository(
        name = "com_github_google_flatbuffers",
        importpath = "github.com/google/flatbuffers",
        sum = "h1:M9dgRyhJemaM4Sw8+66GHBu8ioaQmyPLg1b8VwK5WJg=",
        version = "v23.5.26+incompatible",
    )
    go_repository(
        name = "com_github_google_go_cmp",
        importpath = "github.com/google/go-cmp",
        sum = "h1:ofyhxvXcZhMsU5ulbFiLKl/XBFqE1GSq7atu8tAmTRI=",
        version = "v0.6.0",
    )
    go_repository(
        name = "com_github_google_go_pkcs11",
        importpath = "github.com/google/go-pkcs11",
        sum = "h1:OF1IPgv+F4NmqmJ98KTjdN97Vs1JxDPB3vbmYzV2dpk=",
        version = "v0.2.1-0.20230907215043-c6f79328ddf9",
    )
    go_repository(
        name = "com_github_google_martian_v3",
        importpath = "github.com/google/martian/v3",
        sum = "h1:IqNFLAmvJOgVlpdEBiQbDc2EwKW77amAycfTuWKdfvw=",
        version = "v3.3.2",
    )
    go_repository(
        name = "com_github_google_pprof",
        importpath = "github.com/google/pprof",
        sum = "h1:57oOH2Mu5Nw16KnZAVLdlUjmPH/TSYCKTJgG0OVfX0Y=",
        version = "v0.0.0-20240320155624-b11c3daa6f07",
    )
    go_repository(
        name = "com_github_google_renameio_v2",
        importpath = "github.com/google/renameio/v2",
        sum = "h1:UifI23ZTGY8Tt29JbYFiuyIU3eX+RNFtUwefq9qAhxg=",
        version = "v2.0.0",
    )
    go_repository(
        name = "com_github_google_s2a_go",
        importpath = "github.com/google/s2a-go",
        sum = "h1:60BLSyTrOV4/haCDW4zb1guZItoSq8foHCXrAnjBo/o=",
        version = "v0.1.7",
    )
    go_repository(
        name = "com_github_google_uuid",
        importpath = "github.com/google/uuid",
        sum = "h1:NIvaJDMOsjHA8n1jAhLSgzrAzy1Hgr+hNrb57e+94F0=",
        version = "v1.6.0",
    )
    go_repository(
        name = "com_github_googleapis_enterprise_certificate_proxy",
        importpath = "github.com/googleapis/enterprise-certificate-proxy",
        sum = "h1:Vie5ybvEvT75RniqhfFxPRy3Bf7vr3h0cechB90XaQs=",
        version = "v0.3.2",
    )
    go_repository(
        name = "com_github_googleapis_gax_go_v2",
        build_directives = [
            # See https://github.com/bazelbuild/rules_go/issues/3625#issuecomment-1643804054
            "gazelle:resolve proto proto google/rpc/code.proto @googleapis//google/rpc:code_proto",  # keep
            "gazelle:resolve proto go    google/rpc/code.proto @org_golang_google_genproto_googleapis_rpc//code",  # keep
        ],
        importpath = "github.com/googleapis/gax-go/v2",
        sum = "h1:5/zPPDvw8Q1SuXjrqrZslrqT7dL/uJT2CQii/cLCKqA=",
        version = "v2.12.3",
    )
    go_repository(
        name = "com_github_gorilla_handlers",
        importpath = "github.com/gorilla/handlers",
        sum = "h1:cLTUSsNkgcwhgRqvCNmdbRWG0A3N4F+M2nWKdScwyEE=",
        version = "v1.5.2",
    )
    go_repository(
        name = "com_github_gorilla_mux",
        importpath = "github.com/gorilla/mux",
        sum = "h1:TuBL49tXwgrFYWhqrNgrUNEY92u81SPhu7sTdzQEiWY=",
        version = "v1.8.1",
    )
    go_repository(
        name = "com_github_grpc_ecosystem_grpc_gateway_v2",
        importpath = "github.com/grpc-ecosystem/grpc-gateway/v2",
        sum = "h1:Wqo399gCIufwto+VfwCSvsnfGpF/w5E9CNxSwbpD6No=",
        version = "v2.19.0",
    )
    go_repository(
        name = "com_github_ianlancetaylor_demangle",
        importpath = "github.com/ianlancetaylor/demangle",
        sum = "h1:KwWnWVWCNtNq/ewIX7HIKnELmEx2nDP42yskD/pi7QE=",
        version = "v0.0.0-20240312041847-bd984b5ce465",
    )
    go_repository(
        name = "com_github_inconshreveable_mousetrap",
        importpath = "github.com/inconshreveable/mousetrap",
        sum = "h1:wN+x4NVGpMsO7ErUn/mUI3vEoE6Jt13X2s0bqwp9tc8=",
        version = "v1.1.0",
    )
    go_repository(
        name = "com_github_jmespath_go_jmespath",
        importpath = "github.com/jmespath/go-jmespath",
        sum = "h1:BEgLn5cpjn8UN1mAw4NjwDrS35OdebyEtFe+9YPoQUg=",
        version = "v0.4.0",
    )
    go_repository(
        name = "com_github_johannesboyne_gofakes3",
        importpath = "github.com/johannesboyne/gofakes3",
        sum = "h1:eQGUsj2LcsLzfrHY1noKDSU7h+c9/rw9pQPwbQ9g1jQ=",
        version = "v0.0.0-20221110173912-32fb85c5aed6",
    )
    go_repository(
        name = "com_github_josharian_intern",
        importpath = "github.com/josharian/intern",
        sum = "h1:vlS4z54oSdjm0bgjRigI+G1HpF+tI+9rE5LLzOg8HmY=",
        version = "v1.0.0",
    )
    go_repository(
        name = "com_github_json_iterator_go",
        importpath = "github.com/json-iterator/go",
        sum = "h1:PV8peI4a0ysnczrg+LtxykD8LfKY9ML6u2jnxaEnrnM=",
        version = "v1.1.12",
    )
    go_repository(
        name = "com_github_kisielk_errcheck",
        importpath = "github.com/kisielk/errcheck",
        sum = "h1:e8esj/e4R+SAOwFwN+n3zr0nYeCyeweozKfO23MvHzY=",
        version = "v1.5.0",
    )
    go_repository(
        name = "com_github_kisielk_gotool",
        importpath = "github.com/kisielk/gotool",
        sum = "h1:AV2c/EiW3KqPNT9ZKl07ehoAGi4C5/01Cfbblndcapg=",
        version = "v1.0.0",
    )
    go_repository(
        name = "com_github_klauspost_compress",
        importpath = "github.com/klauspost/compress",
        sum = "h1:d4vBd+7CHydUqpFBgUEKkSdtSugf9YFmSkvUYPquI5E=",
        version = "v1.17.5",
    )
    go_repository(
        name = "com_github_klauspost_cpuid_v2",
        importpath = "github.com/klauspost/cpuid/v2",
        sum = "h1:ndNyv040zDGIDh8thGkXYjnFtiN02M1PVVF+JE/48xc=",
        version = "v2.2.6",
    )
    go_repository(
        name = "com_github_kr_pretty",
        importpath = "github.com/kr/pretty",
        sum = "h1:L/CwN0zerZDmRFUapSPitk6f+Q3+0za1rQkzVuMiMFI=",
        version = "v0.1.0",
    )
    go_repository(
        name = "com_github_kr_pty",
        importpath = "github.com/kr/pty",
        sum = "h1:VkoXIwSboBpnk99O/KFauAEILuNHv5DVFKZMBN/gUgw=",
        version = "v1.1.1",
    )
    go_repository(
        name = "com_github_kr_text",
        importpath = "github.com/kr/text",
        sum = "h1:5Nx0Ya0ZqY2ygV366QzturHI13Jq95ApcVaJBhpS+AY=",
        version = "v0.2.0",
    )
    go_repository(
        name = "com_github_lib_pq",
        importpath = "github.com/lib/pq",
        sum = "h1:YXG7RB+JIjhP29X+OtkiDnYaXQwpS4JEWq7dtCCRUEw=",
        version = "v1.10.9",
    )
    go_repository(
        name = "com_github_linkedin_goavro_v2",
        importpath = "github.com/linkedin/goavro/v2",
        sum = "h1:rIQQSj8jdAUlKQh6DttK8wCRv4t4QO09g1C4aBWXslg=",
        version = "v2.12.0",
    )
    go_repository(
        name = "com_github_lufia_plan9stats",
        importpath = "github.com/lufia/plan9stats",
        sum = "h1:6E+4a0GO5zZEnZ81pIr0yLvtUWk2if982qA3F3QD6H4=",
        version = "v0.0.0-20211012122336-39d0f177ccd0",
    )
    go_repository(
        name = "com_github_magiconair_properties",
        importpath = "github.com/magiconair/properties",
        sum = "h1:IeQXZAiQcpL9mgcAe1Nu6cX9LLw6ExEHKjN0VQdvPDY=",
        version = "v1.8.7",
    )
    go_repository(
        name = "com_github_mailru_easyjson",
        importpath = "github.com/mailru/easyjson",
        sum = "h1:UGYAvKxe3sBsEDzO8ZeWOSlIQfWFlxbzLZe7hwFURr0=",
        version = "v0.7.7",
    )
    go_repository(
        name = "com_github_microsoft_go_winio",
        importpath = "github.com/Microsoft/go-winio",
        sum = "h1:9/kr64B9VUZrLm5YYwbGtUJnMgqWVOdUAXu6Migciow=",
        version = "v0.6.1",
    )
    go_repository(
        name = "com_github_microsoft_hcsshim",
        importpath = "github.com/Microsoft/hcsshim",
        sum = "h1:68vKo2VN8DE9AdN4tnkWnmdhqdbpUFM8OF3Airm7fz8=",
        version = "v0.11.4",
    )
    go_repository(
        name = "com_github_minio_highwayhash",
        importpath = "github.com/minio/highwayhash",
        sum = "h1:Aak5U0nElisjDCfPSG79Tgzkn2gl66NxOMspRrKnA/g=",
        version = "v1.0.2",
    )
    go_repository(
        name = "com_github_moby_patternmatcher",
        importpath = "github.com/moby/patternmatcher",
        sum = "h1:GmP9lR19aU5GqSSFko+5pRqHi+Ohk1O69aFiKkVGiPk=",
        version = "v0.6.0",
    )
    go_repository(
        name = "com_github_moby_sys_sequential",
        importpath = "github.com/moby/sys/sequential",
        sum = "h1:OPvI35Lzn9K04PBbCLW0g4LcFAJgHsvXsRyewg5lXtc=",
        version = "v0.5.0",
    )
    go_repository(
        name = "com_github_moby_sys_user",
        importpath = "github.com/moby/sys/user",
        sum = "h1:WmZ93f5Ux6het5iituh9x2zAG7NFY9Aqi49jjE1PaQg=",
        version = "v0.1.0",
    )
    go_repository(
        name = "com_github_moby_term",
        importpath = "github.com/moby/term",
        sum = "h1:xt8Q1nalod/v7BqbG21f8mQPqH+xAaC9C3N3wfWbVP0=",
        version = "v0.5.0",
    )
    go_repository(
        name = "com_github_modern_go_concurrent",
        importpath = "github.com/modern-go/concurrent",
        sum = "h1:TRLaZ9cD/w8PVh93nsPXa1VrQ6jlwL5oN8l14QlcNfg=",
        version = "v0.0.0-20180306012644-bacd9c7ef1dd",
    )
    go_repository(
        name = "com_github_modern_go_reflect2",
        importpath = "github.com/modern-go/reflect2",
        sum = "h1:xBagoLtFs94CBntxluKeaWgTMpvLxC4ur3nMaC9Gz0M=",
        version = "v1.0.2",
    )
    go_repository(
        name = "com_github_montanaflynn_stats",
        importpath = "github.com/montanaflynn/stats",
        sum = "h1:iruDEfMl2E6fbMZ9s0scYfZQ84/6SPL6zC8ACM2oIL0=",
        version = "v0.0.0-20171201202039-1bf9dbcd8cbe",
    )
    go_repository(
        name = "com_github_morikuni_aec",
        importpath = "github.com/morikuni/aec",
        sum = "h1:nP9CBfwrvYnBRgY6qfDQkygYDmYwOilePFkwzv4dU8A=",
        version = "v1.0.0",
    )
    go_repository(
        name = "com_github_nats_io_jwt_v2",
        importpath = "github.com/nats-io/jwt/v2",
        sum = "h1:/9SWvzc6hTfamcgXJ3uYRpgj+QuY2aLNqRiqrKcrpEo=",
        version = "v2.5.3",
    )
    go_repository(
        name = "com_github_nats_io_nats_go",
        importpath = "github.com/nats-io/nats.go",
        sum = "h1:Bx9BZS+aXYlxW08k8Gd3yR2s73pV5XSoAQUyp1Kwvp0=",
        version = "v1.32.0",
    )
    go_repository(
        name = "com_github_nats_io_nats_server_v2",
        importpath = "github.com/nats-io/nats-server/v2",
        sum = "h1:g1Wd64J5SGsoqWSx1qoNu9/At7a2x+jE7Qtf2XpEx/I=",
        version = "v2.10.10",
    )
    go_repository(
        name = "com_github_nats_io_nkeys",
        importpath = "github.com/nats-io/nkeys",
        sum = "h1:RwNJbbIdYCoClSDNY7QVKZlyb/wfT6ugvFCiKy6vDvI=",
        version = "v0.4.7",
    )
    go_repository(
        name = "com_github_nats_io_nuid",
        importpath = "github.com/nats-io/nuid",
        sum = "h1:5iA8DT8V7q8WK2EScv2padNa/rTESc1KdnPw4TC2paw=",
        version = "v1.0.1",
    )
    go_repository(
        name = "com_github_opencontainers_go_digest",
        importpath = "github.com/opencontainers/go-digest",
        sum = "h1:apOUWs51W5PlhuyGyz9FCeeBIOUDA/6nW8Oi/yOhh5U=",
        version = "v1.0.0",
    )
    go_repository(
        name = "com_github_opencontainers_image_spec",
        importpath = "github.com/opencontainers/image-spec",
        sum = "h1:8SG7/vwALn54lVB/0yZ/MMwhFrPYtpEHQb2IpWsCzug=",
        version = "v1.1.0",
    )
    go_repository(
        name = "com_github_phpdave11_gofpdi",
        importpath = "github.com/phpdave11/gofpdi",
        sum = "h1:o61duiW8M9sMlkVXWlvP92sZJtGKENvW3VExs6dZukQ=",
        version = "v1.0.13",
    )
    go_repository(
        name = "com_github_pierrec_lz4_v4",
        importpath = "github.com/pierrec/lz4/v4",
        sum = "h1:xaKrnTkyoqfh1YItXl56+6KJNVYWlEEPuAQW9xsplYQ=",
        version = "v4.1.18",
    )
    go_repository(
        name = "com_github_pkg_errors",
        importpath = "github.com/pkg/errors",
        sum = "h1:FEBLx1zS214owpjy7qsBeixbURkuhQAwrK5UwLGTwt4=",
        version = "v0.9.1",
    )
    go_repository(
        name = "com_github_pkg_xattr",
        importpath = "github.com/pkg/xattr",
        sum = "h1:5883YPCtkSd8LFbs13nXplj9g9tlrwoJRjgpgMu1/fE=",
        version = "v0.4.9",
    )
    go_repository(
        name = "com_github_pmezard_go_difflib",
        importpath = "github.com/pmezard/go-difflib",
        sum = "h1:4DBwDE0NGyQoBHbLQYPwSUPoCMWR5BEzIk/f1lZbAQM=",
        version = "v1.0.0",
    )
    go_repository(
        name = "com_github_power_devops_perfstat",
        importpath = "github.com/power-devops/perfstat",
        sum = "h1:ncq/mPwQF4JjgDlrVEn3C11VoGHZN7m8qihwgMEtzYw=",
        version = "v0.0.0-20210106213030-5aafc221ea8c",
    )
    go_repository(
        name = "com_github_prometheus_client_model",
        importpath = "github.com/prometheus/client_model",
        sum = "h1:gQz4mCbXsO+nc9n1hCxHcGA3Zx3Eo+UHZoInFGUIXNM=",
        version = "v0.0.0-20190812154241-14fe0d1b01d4",
    )
    go_repository(
        name = "com_github_proullon_ramsql",
        importpath = "github.com/proullon/ramsql",
        sum = "h1:/LRcXJf4lEmhdb4tYcci473I2VynjcZSzh2hsjJ8rSk=",
        version = "v0.1.3",
    )
    go_repository(
        name = "com_github_rogpeppe_clock",
        importpath = "github.com/rogpeppe/clock",
        sum = "h1:3QH7VyOaaiUHNrA9Se4YQIRkDTCw1EJls9xTUCaCeRM=",
        version = "v0.0.0-20190514195947-2896927a307a",
    )
    go_repository(
        name = "com_github_russross_blackfriday",
        importpath = "github.com/russross/blackfriday",
        sum = "h1:KqfZb0pUVN2lYqZUYRddxF4OR8ZMURnJIG5Y3VRLtww=",
        version = "v1.6.0",
    )
    go_repository(
        name = "com_github_ruudk_golang_pdf417",
        importpath = "github.com/ruudk/golang-pdf417",
        sum = "h1:K1Xf3bKttbF+koVGaX5xngRIZ5bVjbmPnaxE/dR08uY=",
        version = "v0.0.0-20201230142125-a7e3863a1245",
    )
    go_repository(
        name = "com_github_ryszard_goskiplist",
        importpath = "github.com/ryszard/goskiplist",
        sum = "h1:GHRpF1pTW19a8tTFrMLUcfWwyC0pnifVo2ClaLq+hP8=",
        version = "v0.0.0-20150312221310-2dfbae5fcf46",
    )
    go_repository(
        name = "com_github_shabbyrobe_gocovmerge",
        importpath = "github.com/shabbyrobe/gocovmerge",
        sum = "h1:J6qvD6rbmOil46orKqJaRPG+zTpoGlBTUdyv8ki63L0=",
        version = "v0.0.0-20180507124511-f6ea450bfb63",
    )
    go_repository(
        name = "com_github_shirou_gopsutil_v3",
        importpath = "github.com/shirou/gopsutil/v3",
        sum = "h1:ZI5bWVeu2ep4/DIxB4U9okeYJ7zp/QLTO4auRb/ty/E=",
        version = "v3.23.9",
    )
    go_repository(
        name = "com_github_shoenig_go_m1cpu",
        importpath = "github.com/shoenig/go-m1cpu",
        sum = "h1:nxdKQNcEB6vzgA2E2bvzKIYRuNj7XNJ4S/aRSwKzFtM=",
        version = "v0.1.6",
    )
    go_repository(
        name = "com_github_sirupsen_logrus",
        importpath = "github.com/sirupsen/logrus",
        sum = "h1:dueUQJ1C2q9oE3F7wvmSGAaVtTmUizReu6fjN8uqzbQ=",
        version = "v1.9.3",
    )
    go_repository(
        name = "com_github_spf13_cobra",
        importpath = "github.com/spf13/cobra",
        sum = "h1:7aJaZx1B85qltLMc546zn58BxxfZdR/W22ej9CFoEf0=",
        version = "v1.8.0",
    )
    go_repository(
        name = "com_github_spf13_pflag",
        importpath = "github.com/spf13/pflag",
        sum = "h1:iy+VFUOCP1a+8yFto/drg2CJ5u0yRoB7fZw3DKv/JXA=",
        version = "v1.0.5",
    )
    go_repository(
        name = "com_github_stretchr_objx",
        importpath = "github.com/stretchr/objx",
        sum = "h1:1zr/of2m5FGMsad5YfcqgdqdWrIhu+EBEJRhR1U7z/c=",
        version = "v0.5.0",
    )
    go_repository(
        name = "com_github_stretchr_testify",
        importpath = "github.com/stretchr/testify",
        sum = "h1:CcVxjf3Q8PM0mHUKJCdn+eZZtm5yQwehR5yeSVQQcUk=",
        version = "v1.8.4",
    )
    go_repository(
        name = "com_github_testcontainers_testcontainers_go",
        importpath = "github.com/testcontainers/testcontainers-go",
        sum = "h1:uqcYdoOHBy1ca7gKODfBd9uTHVK3a7UL848z09MVZ0c=",
        version = "v0.26.0",
    )
    go_repository(
        name = "com_github_tetratelabs_wazero",
        importpath = "github.com/tetratelabs/wazero",
        sum = "h1:z0H1iikCdP8t+q341xqepY4EWvHEw8Es7tlqiVzlP3g=",
        version = "v1.6.0",
    )
    go_repository(
        name = "com_github_tklauser_go_sysconf",
        importpath = "github.com/tklauser/go-sysconf",
        sum = "h1:0QaGUFOdQaIVdPgfITYzaTegZvdCjmYO52cSFAEVmqU=",
        version = "v0.3.12",
    )
    go_repository(
        name = "com_github_tklauser_numcpus",
        importpath = "github.com/tklauser/numcpus",
        sum = "h1:ng9scYS7az0Bk4OZLvrNXNSAO2Pxr1XXRAPyjhIx+Fk=",
        version = "v0.6.1",
    )
    go_repository(
        name = "com_github_xdg_go_pbkdf2",
        importpath = "github.com/xdg-go/pbkdf2",
        sum = "h1:Su7DPu48wXMwC3bs7MCNG+z4FhcyEuz5dlvchbq0B0c=",
        version = "v1.0.0",
    )
    go_repository(
        name = "com_github_xdg_go_scram",
        importpath = "github.com/xdg-go/scram",
        sum = "h1:FHX5I5B4i4hKRVRBCFRxq1iQRej7WO3hhBuJf+UUySY=",
        version = "v1.1.2",
    )
    go_repository(
        name = "com_github_xdg_go_stringprep",
        importpath = "github.com/xdg-go/stringprep",
        sum = "h1:XLI/Ng3O1Atzq0oBs3TWm+5ZVgkq2aqdlvP9JtoZ6c8=",
        version = "v1.0.4",
    )
    go_repository(
        name = "com_github_xeipuuv_gojsonpointer",
        importpath = "github.com/xeipuuv/gojsonpointer",
        sum = "h1:J9EGpcZtP0E/raorCMxlFGSTBrsSlaDGf3jU/qvAE2c=",
        version = "v0.0.0-20180127040702-4e3ac2762d5f",
    )
    go_repository(
        name = "com_github_xeipuuv_gojsonreference",
        importpath = "github.com/xeipuuv/gojsonreference",
        sum = "h1:EzJWgHovont7NscjpAxXsDA8S8BMYve8Y5+7cuRE7R0=",
        version = "v0.0.0-20180127040603-bd5ef7bd5415",
    )
    go_repository(
        name = "com_github_xeipuuv_gojsonschema",
        importpath = "github.com/xeipuuv/gojsonschema",
        sum = "h1:LhYJRs+L4fBtjZUfuSZIKGeVu0QRy8e5Xi7D17UxZ74=",
        version = "v1.2.0",
    )
    go_repository(
        name = "com_github_xitongsys_parquet_go",
        importpath = "github.com/xitongsys/parquet-go",
        sum = "h1:MhCaXii4eqceKPu9BwrjLqyK10oX9WF+xGhwvwbw7xM=",
        version = "v1.6.2",
    )
    go_repository(
        name = "com_github_xitongsys_parquet_go_source",
        importpath = "github.com/xitongsys/parquet-go-source",
        sum = "h1:UDtocVeACpnwauljUbeHD9UOjjcvF5kLUHruww7VT9A=",
        version = "v0.0.0-20220315005136-aec0fe3e777c",
    )
    go_repository(
        name = "com_github_youmark_pkcs8",
        importpath = "github.com/youmark/pkcs8",
        sum = "h1:splanxYIlg+5LfHAM6xpdFEAYOk8iySO56hMFq6uLyA=",
        version = "v0.0.0-20181117223130-1be2e3e5546d",
    )
    go_repository(
        name = "com_github_yuin_goldmark",
        importpath = "github.com/yuin/goldmark",
        sum = "h1:fVcFKWvrslecOb/tg+Cc05dkeYx540o0FuFt3nUVDoE=",
        version = "v1.4.13",
    )
    go_repository(
        name = "com_github_yusufpapurcu_wmi",
        importpath = "github.com/yusufpapurcu/wmi",
        sum = "h1:E1ctvB7uKFMOJw3fdOW32DwGE9I7t++CRUEMKvFoFiw=",
        version = "v1.2.3",
    )
    go_repository(
        name = "com_github_zeebo_xxh3",
        importpath = "github.com/zeebo/xxh3",
        sum = "h1:xZmwmqxHZA8AI603jOQ0tMqmBr9lPeFwGg6d+xy9DC0=",
        version = "v1.0.2",
    )
    go_repository(
        name = "com_google_cloud_go",
        importpath = "cloud.google.com/go",
        sum = "h1:uJSeirPke5UNZHIb4SxfZklVSiWWVqW4oXlETwZziwM=",
        version = "v0.112.1",
    )
    go_repository(
        name = "com_google_cloud_go_accessapproval",
        importpath = "cloud.google.com/go/accessapproval",
        sum = "h1:fMbP4cJX/926h+kwGtABmcG83PXsjkB+q7nSBzZpJoo=",
        version = "v1.7.6",
    )
    go_repository(
        name = "com_google_cloud_go_accesscontextmanager",
        importpath = "cloud.google.com/go/accesscontextmanager",
        sum = "h1:NipmPd3BCzwa/mr40SK8pWRkbzv9Th5Azhi4dBYazlM=",
        version = "v1.8.6",
    )
    go_repository(
        name = "com_google_cloud_go_aiplatform",
        importpath = "cloud.google.com/go/aiplatform",
        sum = "h1:4X8As3ZDAg71d8l1f6vI/09MaIalXSMZ0nDrLY1wmbo=",
        version = "v1.64.0",
    )
    go_repository(
        name = "com_google_cloud_go_analytics",
        importpath = "cloud.google.com/go/analytics",
        sum = "h1:UH/PWBcPxHKolWxaS3hO+aj+wDTuq7XKvdtpqR3lyOI=",
        version = "v0.23.1",
    )
    go_repository(
        name = "com_google_cloud_go_apigateway",
        importpath = "cloud.google.com/go/apigateway",
        sum = "h1:60GMRN1JFwq9MldvEVMdR3gDJ0vI0C/BwgkImG6bx/M=",
        version = "v1.6.6",
    )
    go_repository(
        name = "com_google_cloud_go_apigeeconnect",
        importpath = "cloud.google.com/go/apigeeconnect",
        sum = "h1:ObsKNGtdu0ckkCSUpCN5fVAVg+DmzFg89JlqsW4OAWk=",
        version = "v1.6.6",
    )
    go_repository(
        name = "com_google_cloud_go_apigeeregistry",
        importpath = "cloud.google.com/go/apigeeregistry",
        sum = "h1:l8VFHdNMC+9Q4EHKye2eOZBu5IwddXF6ufAXI7D+PB8=",
        version = "v0.8.4",
    )
    go_repository(
        name = "com_google_cloud_go_appengine",
        importpath = "cloud.google.com/go/appengine",
        sum = "h1:cM+Lj0A0nCtujM9lqRId5L8hz7bHRfu3q3KdKtpn+38=",
        version = "v1.8.6",
    )
    go_repository(
        name = "com_google_cloud_go_area120",
        importpath = "cloud.google.com/go/area120",
        sum = "h1:7QJ4ZzqLOwh0pHD3UAa+VwiyWmDI7bdmkYKVkte8/wg=",
        version = "v0.8.6",
    )
    go_repository(
        name = "com_google_cloud_go_artifactregistry",
        importpath = "cloud.google.com/go/artifactregistry",
        sum = "h1:icIyRzJ1Ag6EOafuDuFFJ/AdStcOFRVfSGURn27/7Pk=",
        version = "v1.14.8",
    )
    go_repository(
        name = "com_google_cloud_go_asset",
        importpath = "cloud.google.com/go/asset",
        sum = "h1:+NpxL5L53VY91EoJTHeGGXSWEUllf2hhXpCyTnSrd3Q=",
        version = "v1.18.1",
    )
    go_repository(
        name = "com_google_cloud_go_assuredworkloads",
        importpath = "cloud.google.com/go/assuredworkloads",
        sum = "h1:3NlUes0xLN2kcSU24qQADFYsOaetCPg0HSA302AyV5s=",
        version = "v1.11.6",
    )
    go_repository(
        name = "com_google_cloud_go_automl",
        importpath = "cloud.google.com/go/automl",
        sum = "h1:NHBO5cjo2IgwaJ5qlez/iA35XI1db87PPlOB0Kjt5RM=",
        version = "v1.13.6",
    )
    go_repository(
        name = "com_google_cloud_go_baremetalsolution",
        importpath = "cloud.google.com/go/baremetalsolution",
        sum = "h1:jCR4rnVsG6ocK6ngFr2Z6ugKZfTENmMZkiV6Ma2tEeE=",
        version = "v1.2.5",
    )
    go_repository(
        name = "com_google_cloud_go_batch",
        importpath = "cloud.google.com/go/batch",
        sum = "h1:b9fVZDxxp4LWMhXV7uAhyMGmPuzlzPrsZ0uh+RM92h8=",
        version = "v1.8.3",
    )
    go_repository(
        name = "com_google_cloud_go_beyondcorp",
        importpath = "cloud.google.com/go/beyondcorp",
        sum = "h1:fnil8viEdcAJJiwiJPCT2fl3Grx3XFwXxTq7n9g/8QM=",
        version = "v1.0.5",
    )
    go_repository(
        name = "com_google_cloud_go_bigquery",
        importpath = "cloud.google.com/go/bigquery",
        sum = "h1:CpT+/njKuKT3CEmswm6IbhNu9u35zt5dO4yPDLW+nG4=",
        version = "v1.59.1",
    )
    go_repository(
        name = "com_google_cloud_go_bigtable",
        importpath = "cloud.google.com/go/bigtable",
        sum = "h1:BFN4jhkA9ULYYV2Ug7AeOtetVLnN2jKuIq5TcRc5C38=",
        version = "v1.21.0",
    )
    go_repository(
        name = "com_google_cloud_go_billing",
        importpath = "cloud.google.com/go/billing",
        sum = "h1:XcYB8aKj97d4/0kh+LQgrxPxOo/0jgPNp5Q1nyzCyvs=",
        version = "v1.18.4",
    )
    go_repository(
        name = "com_google_cloud_go_binaryauthorization",
        importpath = "cloud.google.com/go/binaryauthorization",
        sum = "h1:XiAdW5HAWtk9WEjEA5MXYiRJ28Tjp1xGPPAMRFI5bnc=",
        version = "v1.8.2",
    )
    go_repository(
        name = "com_google_cloud_go_certificatemanager",
        importpath = "cloud.google.com/go/certificatemanager",
        sum = "h1:oc15T+leZ2Wm5oocvGTbDXwonka0chpJTEiHIVsBiEA=",
        version = "v1.8.0",
    )
    go_repository(
        name = "com_google_cloud_go_channel",
        importpath = "cloud.google.com/go/channel",
        sum = "h1:rBnTls9G5nC/jOrb9V/tZFHFXt6kBMNlIQKg6DjAlRM=",
        version = "v1.17.6",
    )
    go_repository(
        name = "com_google_cloud_go_cloudbuild",
        importpath = "cloud.google.com/go/cloudbuild",
        sum = "h1:66hY1gXV2cdn4gquy5TieaJwaZmRzrQ5cK++pVgnCkQ=",
        version = "v1.16.0",
    )
    go_repository(
        name = "com_google_cloud_go_clouddms",
        importpath = "cloud.google.com/go/clouddms",
        sum = "h1:t1nc49kRzEU2vrDxQQIEc5rZ4Hr187YrdOZPMAgMwMI=",
        version = "v1.7.5",
    )
    go_repository(
        name = "com_google_cloud_go_cloudtasks",
        importpath = "cloud.google.com/go/cloudtasks",
        sum = "h1:Ev+poxwb7pudBhiF0ObwAWT7Dh9BZAcsvAfFTWg0MPc=",
        version = "v1.12.7",
    )
    go_repository(
        name = "com_google_cloud_go_compute",
        build_directives = [
            # See https://github.com/bazelbuild/rules_go/issues/1877#issuecomment-1874616324
            "gazelle:resolve go google.golang.org/genproto/googleapis/api/annotations @org_golang_google_genproto//googleapis/api/annotations",  # keep
            "gazelle:resolve go google.golang.org/genproto/googleapis/api @org_golang_google_genproto//googleapis/api",  # keep
        ],
        importpath = "cloud.google.com/go/compute",
        sum = "h1:ZRpHJedLtTpKgr3RV1Fx23NuaAEN1Zfx9hw1u4aJdjU=",
        version = "v1.25.1",
    )
    go_repository(
        name = "com_google_cloud_go_compute_metadata",
        importpath = "cloud.google.com/go/compute/metadata",
        sum = "h1:mg4jlk7mCAj6xXp9UJ4fjI9VUI5rubuGBW5aJ7UnBMY=",
        version = "v0.2.3",
    )
    go_repository(
        name = "com_google_cloud_go_contactcenterinsights",
        importpath = "cloud.google.com/go/contactcenterinsights",
        sum = "h1:sCDKUmDj9Tfd6Qj7x4XbwC43oYzEBwSDLC1tReQWS/Y=",
        version = "v1.13.1",
    )
    go_repository(
        name = "com_google_cloud_go_container",
        importpath = "cloud.google.com/go/container",
        sum = "h1:0yEtOgltMArvXSI32Ju4mLqsboUeIWjbon5rIRBV8UA=",
        version = "v1.33.1",
    )
    go_repository(
        name = "com_google_cloud_go_containeranalysis",
        importpath = "cloud.google.com/go/containeranalysis",
        sum = "h1:yzohQ0HDoZq2TtCJkbUBsJs9RIR5WbKZlHrD7ilp2yg=",
        version = "v0.11.5",
    )
    go_repository(
        name = "com_google_cloud_go_datacatalog",
        importpath = "cloud.google.com/go/datacatalog",
        sum = "h1:BGDsEjqpAo0Ka+b9yDLXnE5k+jU3lXGMh//NsEeDMIg=",
        version = "v1.20.0",
    )
    go_repository(
        name = "com_google_cloud_go_dataflow",
        importpath = "cloud.google.com/go/dataflow",
        sum = "h1:GuZJgkOL64cYySwYEYqQkggdxwoZTk8cvekQW0t3KRM=",
        version = "v0.9.6",
    )
    go_repository(
        name = "com_google_cloud_go_dataform",
        importpath = "cloud.google.com/go/dataform",
        sum = "h1:0EzWf+c2R5V/ujZBb22H/EL5wpzD/bDFYPA2f3czB1g=",
        version = "v0.9.3",
    )
    go_repository(
        name = "com_google_cloud_go_datafusion",
        importpath = "cloud.google.com/go/datafusion",
        sum = "h1:zSmMj/qZ0Yk+q/v5Wg40FTJ0WYPCtanYYekRt7cSKJ0=",
        version = "v1.7.6",
    )
    go_repository(
        name = "com_google_cloud_go_datalabeling",
        importpath = "cloud.google.com/go/datalabeling",
        sum = "h1:2zz44bPbDMHsPanQ89SybqhHDQBH1EZk8icGotyrvSU=",
        version = "v0.8.6",
    )
    go_repository(
        name = "com_google_cloud_go_dataplex",
        importpath = "cloud.google.com/go/dataplex",
        sum = "h1:Ob8NPT1UcB4kDaDx7/UdsRfZ8xUvUggZshXUlGWDahk=",
        version = "v1.15.0",
    )
    go_repository(
        name = "com_google_cloud_go_dataproc_v2",
        importpath = "cloud.google.com/go/dataproc/v2",
        sum = "h1:+cM8p/R6FdTuQYlriJOSUCvAZfMDgBKf0/ph9bMIjaY=",
        version = "v2.4.1",
    )
    go_repository(
        name = "com_google_cloud_go_dataqna",
        importpath = "cloud.google.com/go/dataqna",
        sum = "h1:FI/1q7VnANchQR9ug+nzujfiusLMfC3BHT7/fHi+BVU=",
        version = "v0.8.6",
    )
    go_repository(
        name = "com_google_cloud_go_datastore",
        importpath = "cloud.google.com/go/datastore",
        sum = "h1:0P9WcsQeTWjuD1H14JIY7XQscIPQ4Laje8ti96IC5vg=",
        version = "v1.15.0",
    )
    go_repository(
        name = "com_google_cloud_go_datastream",
        importpath = "cloud.google.com/go/datastream",
        sum = "h1:nHdOKbFmKJ4tPjGtNNIO0//G7QAht6eHTUnREWPn2yI=",
        version = "v1.10.5",
    )
    go_repository(
        name = "com_google_cloud_go_deploy",
        importpath = "cloud.google.com/go/deploy",
        sum = "h1:UxcxzjwxGPkT7RBdMmoc5a7QxHQVdpZllD6el2VC3JA=",
        version = "v1.17.2",
    )
    go_repository(
        name = "com_google_cloud_go_dialogflow",
        importpath = "cloud.google.com/go/dialogflow",
        sum = "h1:MGdOD6ZqXkburW/Ij0cUe1ArO9PLHuWLWclw2IHt2go=",
        version = "v1.51.0",
    )
    go_repository(
        name = "com_google_cloud_go_dlp",
        importpath = "cloud.google.com/go/dlp",
        sum = "h1:dTsEN6r1BoplUACz7teOmE6lRG1swREiwXkfkY7bi6c=",
        version = "v1.12.1",
    )
    go_repository(
        name = "com_google_cloud_go_documentai",
        importpath = "cloud.google.com/go/documentai",
        sum = "h1:UdDy7nDTwr+mN1KiJqsj5AabauoW9SkgH9eY8BuFXJE=",
        version = "v1.26.1",
    )
    go_repository(
        name = "com_google_cloud_go_domains",
        importpath = "cloud.google.com/go/domains",
        sum = "h1:NHqZk4XzHFlmXM3LMGwDVET4NKr60W2jaNCRGYod5Ic=",
        version = "v0.9.6",
    )
    go_repository(
        name = "com_google_cloud_go_edgecontainer",
        importpath = "cloud.google.com/go/edgecontainer",
        sum = "h1:FUhibxnkl9NFs+nZw3D1+L+vxkV2mbRZCDtexlaZdQw=",
        version = "v1.1.6",
    )
    go_repository(
        name = "com_google_cloud_go_errorreporting",
        importpath = "cloud.google.com/go/errorreporting",
        sum = "h1:kj1XEWMu8P0qlLhm3FwcaFsUvXChV/OraZwA70trRR0=",
        version = "v0.3.0",
    )
    go_repository(
        name = "com_google_cloud_go_essentialcontacts",
        importpath = "cloud.google.com/go/essentialcontacts",
        sum = "h1:FDdJGJEXK4RxvT6gdRBqGaCQVpi96RRB7MTyRHUcb20=",
        version = "v1.6.7",
    )
    go_repository(
        name = "com_google_cloud_go_eventarc",
        importpath = "cloud.google.com/go/eventarc",
        sum = "h1:JMUiLYzxkxr7BqnCPkyJ6Ycgrs6YQlZT44H0rHg7jQY=",
        version = "v1.13.5",
    )
    go_repository(
        name = "com_google_cloud_go_filestore",
        importpath = "cloud.google.com/go/filestore",
        sum = "h1:BpaB7bxICPUTntAV+yVUK9bxAUOv7uHRSEibSKMBJVA=",
        version = "v1.8.2",
    )
    go_repository(
        name = "com_google_cloud_go_firestore",
        importpath = "cloud.google.com/go/firestore",
        sum = "h1:/k8ppuWOtNuDHt2tsRV42yI21uaGnKDEQnRFeBpbFF8=",
        version = "v1.15.0",
    )
    go_repository(
        name = "com_google_cloud_go_functions",
        importpath = "cloud.google.com/go/functions",
        sum = "h1:0kcko/2AKwm4USnWcGs/W/k++PAYPA3dYaQw1y5Xg3M=",
        version = "v1.16.1",
    )
    go_repository(
        name = "com_google_cloud_go_gkebackup",
        importpath = "cloud.google.com/go/gkebackup",
        sum = "h1:ZSFtQdDJhOc/6cBfTEDF5yVZMBsbecjoq8jc+oACqKw=",
        version = "v1.3.6",
    )
    go_repository(
        name = "com_google_cloud_go_gkeconnect",
        importpath = "cloud.google.com/go/gkeconnect",
        sum = "h1:7X9P6lGkOF/nJRYZBQBG2XPhunqWbNMacy9AXN7qUcU=",
        version = "v0.8.6",
    )
    go_repository(
        name = "com_google_cloud_go_gkehub",
        importpath = "cloud.google.com/go/gkehub",
        sum = "h1:kKreFf+097KfW+Tz/SqZKeXs/eFOjs1NDrsVjRPI18o=",
        version = "v0.14.6",
    )
    go_repository(
        name = "com_google_cloud_go_gkemulticloud",
        importpath = "cloud.google.com/go/gkemulticloud",
        sum = "h1:CFBoDcQi9zLOkzM6xqmRzljZhF4A6A47QaQ0WtNd+DA=",
        version = "v1.1.2",
    )
    go_repository(
        name = "com_google_cloud_go_gsuiteaddons",
        importpath = "cloud.google.com/go/gsuiteaddons",
        sum = "h1:q3x2NE0je/tSVL66MAht5YVbGGHjTV9BxFD2lyDQ0dU=",
        version = "v1.6.6",
    )
    go_repository(
        name = "com_google_cloud_go_iam",
        build_directives = [
            # See https://github.com/bazelbuild/rules_go/issues/1877#issuecomment-1874616324
            "gazelle:resolve go google.golang.org/genproto/googleapis/api/annotations @org_golang_google_genproto//googleapis/api/annotations",  # keep
            "gazelle:resolve go google.golang.org/genproto/googleapis/api @org_golang_google_genproto//googleapis/api",  # keep
        ],
        importpath = "cloud.google.com/go/iam",
        sum = "h1:z4VHOhwKLF/+UYXAJDFwGtNF0b6gjsW1Pk9Ml0U/IoM=",
        version = "v1.1.7",
    )
    go_repository(
        name = "com_google_cloud_go_iap",
        importpath = "cloud.google.com/go/iap",
        sum = "h1:FrLAtgXzWPwe8rNp7AD+2Lgg4LqyhgXvEdiGK+jtd9g=",
        version = "v1.9.5",
    )
    go_repository(
        name = "com_google_cloud_go_ids",
        importpath = "cloud.google.com/go/ids",
        sum = "h1:tNc3NpIp2LUmFJxP2CBlzYw0FTnd68r73mIzg8UlM3Q=",
        version = "v1.4.6",
    )
    go_repository(
        name = "com_google_cloud_go_iot",
        importpath = "cloud.google.com/go/iot",
        sum = "h1:nRV/e1e3lEjsVoD5mW99JERwL8MKohyQyY63+lfBMA4=",
        version = "v1.7.6",
    )
    go_repository(
        name = "com_google_cloud_go_kms",
        build_directives = [
            # See https://github.com/bazelbuild/rules_go/issues/1877#issuecomment-1874616324
            "gazelle:resolve go google.golang.org/genproto/googleapis/api/annotations @org_golang_google_genproto//googleapis/api/annotations",  # keep
            "gazelle:resolve go google.golang.org/genproto/googleapis/api @org_golang_google_genproto//googleapis/api",  # keep
        ],
        importpath = "cloud.google.com/go/kms",
        sum = "h1:szIeDCowID8th2i8XE4uRev5PMxQFqW+JjwYxL9h6xs=",
        version = "v1.15.8",
    )
    go_repository(
        name = "com_google_cloud_go_language",
        importpath = "cloud.google.com/go/language",
        sum = "h1:srkreCxnVa5+a5PXUri/K+VWxG50wGvz5+PEYjgaENQ=",
        version = "v1.12.4",
    )
    go_repository(
        name = "com_google_cloud_go_lifesciences",
        importpath = "cloud.google.com/go/lifesciences",
        sum = "h1:8w3edjRiSN6GCxT0uJoXr6Zo2XNYD+6TxPZa7uIIOaU=",
        version = "v0.9.6",
    )
    go_repository(
        name = "com_google_cloud_go_logging",
        importpath = "cloud.google.com/go/logging",
        sum = "h1:iEIOXFO9EmSiTjDmfpbRjOxECO7R8C7b8IXUGOj7xZw=",
        version = "v1.9.0",
    )
    go_repository(
        name = "com_google_cloud_go_longrunning",
        importpath = "cloud.google.com/go/longrunning",
        sum = "h1:xAe8+0YaWoCKr9t1+aWe+OeQgN/iJK1fEgZSXmjuEaE=",
        version = "v0.5.6",
    )
    go_repository(
        name = "com_google_cloud_go_managedidentities",
        importpath = "cloud.google.com/go/managedidentities",
        sum = "h1:7+hGPQSojhnYNZCg3fG2mQIF7XMfvNpCpi2Zg5/Qx1g=",
        version = "v1.6.6",
    )
    go_repository(
        name = "com_google_cloud_go_maps",
        importpath = "cloud.google.com/go/maps",
        sum = "h1:vcqmqk0wt1NRzQc84Qo6z8HYyol/znqbG7tAS5Qm91g=",
        version = "v1.7.1",
    )
    go_repository(
        name = "com_google_cloud_go_mediatranslation",
        importpath = "cloud.google.com/go/mediatranslation",
        sum = "h1:EVW0wCQ7asoMjw8fm8oUe6pQWBaVQth/iquk7ysidy0=",
        version = "v0.8.6",
    )
    go_repository(
        name = "com_google_cloud_go_memcache",
        importpath = "cloud.google.com/go/memcache",
        sum = "h1:rqDPCIUfVBvv7ojOGx5PRkPgWeWSKpOht2w6plaxklY=",
        version = "v1.10.6",
    )
    go_repository(
        name = "com_google_cloud_go_metastore",
        importpath = "cloud.google.com/go/metastore",
        sum = "h1:K7gyYoqPvQgCc82tiB0CQkXOpg8AZxJtRGMVdN5B83U=",
        version = "v1.13.5",
    )
    go_repository(
        name = "com_google_cloud_go_monitoring",
        build_directives = [
            # See https://github.com/bazelbuild/rules_go/issues/1877#issuecomment-1874616324
            "gazelle:resolve go google.golang.org/genproto/googleapis/api/annotations @org_golang_google_genproto//googleapis/api/annotations",  # keep
            "gazelle:resolve go google.golang.org/genproto/googleapis/api @org_golang_google_genproto//googleapis/api",  # keep
        ],
        importpath = "cloud.google.com/go/monitoring",
        sum = "h1:0yvFXK+xQd95VKo6thndjwnJMno7c7Xw1CwMByg0B+8=",
        version = "v1.18.1",
    )
    go_repository(
        name = "com_google_cloud_go_networkconnectivity",
        importpath = "cloud.google.com/go/networkconnectivity",
        sum = "h1:t67aEKwmO+SXvQC5ncOjm3vTwnsbO/mTzlCWdK0nwqs=",
        version = "v1.14.5",
    )
    go_repository(
        name = "com_google_cloud_go_networkmanagement",
        importpath = "cloud.google.com/go/networkmanagement",
        sum = "h1:ZWn7dZix5/5KPFZslxjmh44Q6VSOXM16EAUHPf6Py/8=",
        version = "v1.11.0",
    )
    go_repository(
        name = "com_google_cloud_go_networksecurity",
        importpath = "cloud.google.com/go/networksecurity",
        sum = "h1:3ggPKshcFs1oRh5qI+Gq1s2CIU9BL99MKtYSBG4Z8s0=",
        version = "v0.9.6",
    )
    go_repository(
        name = "com_google_cloud_go_notebooks",
        importpath = "cloud.google.com/go/notebooks",
        sum = "h1:A9jxIdxEccgL9iJLqvU4j5HT3/13YluoF2IbiV+hAN4=",
        version = "v1.11.4",
    )
    go_repository(
        name = "com_google_cloud_go_optimization",
        importpath = "cloud.google.com/go/optimization",
        sum = "h1:T/j8xyIkmHGjU6kxeUjK3UTqiXlbvpZQ2A+F5vnH21Y=",
        version = "v1.6.4",
    )
    go_repository(
        name = "com_google_cloud_go_orchestration",
        importpath = "cloud.google.com/go/orchestration",
        sum = "h1:i5iSxsu1Cx1itTQEEY/YvsAo1OO8gosGGXhnOjBjgJA=",
        version = "v1.9.1",
    )
    go_repository(
        name = "com_google_cloud_go_orgpolicy",
        importpath = "cloud.google.com/go/orgpolicy",
        sum = "h1:x9GttuUZXXeKcJgHSGxYoPn2hOJhhuaN5YYJKfAfmLo=",
        version = "v1.12.2",
    )
    go_repository(
        name = "com_google_cloud_go_osconfig",
        importpath = "cloud.google.com/go/osconfig",
        sum = "h1:wIOhgzklE0hHZsho02rRVXYBHSfsAwYZYIaxFaUBIjs=",
        version = "v1.12.6",
    )
    go_repository(
        name = "com_google_cloud_go_oslogin",
        importpath = "cloud.google.com/go/oslogin",
        sum = "h1:v71OrrkKyqr5Mfnt345GqSOURzByv08qfrtvfhOVcnc=",
        version = "v1.13.2",
    )
    go_repository(
        name = "com_google_cloud_go_phishingprotection",
        importpath = "cloud.google.com/go/phishingprotection",
        sum = "h1:DcAre1psFwJM/FBA/MkDj0H6uxZhACE5IW/xF9ssHDQ=",
        version = "v0.8.6",
    )
    go_repository(
        name = "com_google_cloud_go_policytroubleshooter",
        importpath = "cloud.google.com/go/policytroubleshooter",
        sum = "h1:wxBRfNoMy7rnoEeaTOHIEHCUEdUIQIwQGUqfBWH6cyQ=",
        version = "v1.10.4",
    )
    go_repository(
        name = "com_google_cloud_go_privatecatalog",
        importpath = "cloud.google.com/go/privatecatalog",
        sum = "h1:bcIABOUmpnzQip83OVv+Ju/NxXjUTRLUSP+FVLFG6kk=",
        version = "v0.9.6",
    )
    go_repository(
        name = "com_google_cloud_go_profiler",
        importpath = "cloud.google.com/go/profiler",
        sum = "h1:ZeRDZbsOBDyRG0OiK0Op1/XWZ3xeLwJc9zjkzczUxyY=",
        version = "v0.4.0",
    )
    go_repository(
        name = "com_google_cloud_go_pubsub",
        importpath = "cloud.google.com/go/pubsub",
        sum = "h1:0uEEfaB1VIJzabPpwpZf44zWAKAme3zwKKxHk7vJQxQ=",
        version = "v1.37.0",
    )
    go_repository(
        name = "com_google_cloud_go_pubsublite",
        importpath = "cloud.google.com/go/pubsublite",
        sum = "h1:pX+idpWMIH30/K7c0epN6V703xpIcMXWRjKJsz0tYGY=",
        version = "v1.8.1",
    )
    go_repository(
        name = "com_google_cloud_go_recaptchaenterprise_v2",
        importpath = "cloud.google.com/go/recaptchaenterprise/v2",
        sum = "h1:TI9XcWccMeipi4CFb8tJ9RvFgt9+f4W00t8RUNoXiPk=",
        version = "v2.11.1",
    )
    go_repository(
        name = "com_google_cloud_go_recommendationengine",
        importpath = "cloud.google.com/go/recommendationengine",
        sum = "h1:m0eQtYCToxMSbDKOnpJ2YGdQhyjOPffg4Y8lM2RWzao=",
        version = "v0.8.6",
    )
    go_repository(
        name = "com_google_cloud_go_recommender",
        importpath = "cloud.google.com/go/recommender",
        sum = "h1:3M6lD39/GlOMYOikeF5wflSa4EP5pGFthoIASbyhIXE=",
        version = "v1.12.2",
    )
    go_repository(
        name = "com_google_cloud_go_redis",
        importpath = "cloud.google.com/go/redis",
        sum = "h1:zlGxeAsiwcPU+Cta76ALduhdBAVhuYpEjv59V5L/ves=",
        version = "v1.14.3",
    )
    go_repository(
        name = "com_google_cloud_go_resourcemanager",
        importpath = "cloud.google.com/go/resourcemanager",
        sum = "h1:VPfJFbWxrTYQzEXCDbJNpcvSB8eZhTSM0YHH146fIB8=",
        version = "v1.9.6",
    )
    go_repository(
        name = "com_google_cloud_go_resourcesettings",
        importpath = "cloud.google.com/go/resourcesettings",
        sum = "h1:l/IbRDDmGJFlR4bRZGtfYvix1Pu0jAKGLr7wgUtixHQ=",
        version = "v1.6.6",
    )
    go_repository(
        name = "com_google_cloud_go_retail",
        importpath = "cloud.google.com/go/retail",
        sum = "h1:AyVdElkdIU3JedWpX/qENbt8iUmKD+kiyj7ZpzguhTg=",
        version = "v1.16.1",
    )
    go_repository(
        name = "com_google_cloud_go_run",
        importpath = "cloud.google.com/go/run",
        sum = "h1:xQND6EJn1LgouCLPSfykkzagyr4gq4FKiRexNxXixV0=",
        version = "v1.3.6",
    )
    go_repository(
        name = "com_google_cloud_go_scheduler",
        importpath = "cloud.google.com/go/scheduler",
        sum = "h1:h1/VZk0XdkSh/jI7dDNp3V0Qi8yTkclOljDVPelXvAw=",
        version = "v1.10.7",
    )
    go_repository(
        name = "com_google_cloud_go_secretmanager",
        importpath = "cloud.google.com/go/secretmanager",
        sum = "h1:e5pIo/QEgiFiHPVJPxM5jbtUr4O/u5h2zLHYtkFQr24=",
        version = "v1.12.0",
    )
    go_repository(
        name = "com_google_cloud_go_security",
        importpath = "cloud.google.com/go/security",
        sum = "h1:LYMj7ISEEjVQ0ub6E6ygGhjVbNQTH5CawKZz0bbPMVE=",
        version = "v1.15.6",
    )
    go_repository(
        name = "com_google_cloud_go_securitycenter",
        importpath = "cloud.google.com/go/securitycenter",
        sum = "h1:NpEJeFbm3ad3ibpbpIBKXJS7eQq1cZhtt9nrDTMO/QQ=",
        version = "v1.28.0",
    )
    go_repository(
        name = "com_google_cloud_go_servicedirectory",
        importpath = "cloud.google.com/go/servicedirectory",
        sum = "h1:gkzx9Cd+OTOD+zY4u5vtbdvOx7vrvHYdeDiNdC6vKyw=",
        version = "v1.11.5",
    )
    go_repository(
        name = "com_google_cloud_go_shell",
        importpath = "cloud.google.com/go/shell",
        sum = "h1:/oJf9sboa2FfHWCmHXy+XfTRnZy8lC7O5zFyfE1EA6s=",
        version = "v1.7.6",
    )
    go_repository(
        name = "com_google_cloud_go_spanner",
        importpath = "cloud.google.com/go/spanner",
        sum = "h1:O9kf49dfaDRzPpKJNChHUJ+Bao02WPedZb8ZPyi02lI=",
        version = "v1.60.0",
    )
    go_repository(
        name = "com_google_cloud_go_speech",
        importpath = "cloud.google.com/go/speech",
        sum = "h1:xo/cmhBtqoqqNg/5I8m0ECXPiqYg2fS2ioOccH+qbKE=",
        version = "v1.22.1",
    )
    go_repository(
        name = "com_google_cloud_go_storage",
        build_directives = [
            # See https://github.com/bazelbuild/rules_go/issues/1877#issuecomment-1874616324
            "gazelle:resolve go google.golang.org/genproto/googleapis/api/annotations @org_golang_google_genproto//googleapis/api/annotations",  # keep
            "gazelle:resolve go google.golang.org/genproto/googleapis/api @org_golang_google_genproto//googleapis/api",  # keep
        ],
        importpath = "cloud.google.com/go/storage",
        sum = "h1:MvraqHKhogCOTXTlct/9C3K3+Uy2jBmFYb3/Sp6dVtY=",
        version = "v1.39.1",
    )
    go_repository(
        name = "com_google_cloud_go_storagetransfer",
        importpath = "cloud.google.com/go/storagetransfer",
        sum = "h1:BawJo/u0P21cdxc2gB878qIFDC80COq2i0qWZeNevSw=",
        version = "v1.10.5",
    )
    go_repository(
        name = "com_google_cloud_go_talent",
        importpath = "cloud.google.com/go/talent",
        sum = "h1:4xgDFfOcgcSY0dUzaSc2tQCSRoLDEJ5CfbW5jfcgNJk=",
        version = "v1.6.7",
    )
    go_repository(
        name = "com_google_cloud_go_texttospeech",
        importpath = "cloud.google.com/go/texttospeech",
        sum = "h1:gLEyDoJeFGdoX7jSKbf+nJy7CTgjsSbCZXwzzkXgH9w=",
        version = "v1.7.6",
    )
    go_repository(
        name = "com_google_cloud_go_tpu",
        importpath = "cloud.google.com/go/tpu",
        sum = "h1:Cb1txkZYbKlGIZ4tQr9EjEB9snAOU6qyjvNezGXDunI=",
        version = "v1.6.6",
    )
    go_repository(
        name = "com_google_cloud_go_trace",
        importpath = "cloud.google.com/go/trace",
        sum = "h1:XF0Ejdw0NpRfAvuZUeQe3ClAG4R/9w5JYICo7l2weaw=",
        version = "v1.10.6",
    )
    go_repository(
        name = "com_google_cloud_go_translate",
        importpath = "cloud.google.com/go/translate",
        sum = "h1:SXOtKYnT7ZkeMtPwujaBOBt5Ph4kf6LIuMpAgu/WON0=",
        version = "v1.10.2",
    )
    go_repository(
        name = "com_google_cloud_go_video",
        importpath = "cloud.google.com/go/video",
        sum = "h1:y4jgUqDiWMfX+beJnlrnloBQxEIa9v+KrlkD2QJVpeE=",
        version = "v1.20.5",
    )
    go_repository(
        name = "com_google_cloud_go_videointelligence",
        importpath = "cloud.google.com/go/videointelligence",
        sum = "h1:P0Sa8+5KOEAVk/fazUNjVPzRCijCheZWJ8wL8xBn9Uk=",
        version = "v1.11.6",
    )
    go_repository(
        name = "com_google_cloud_go_vision_v2",
        importpath = "cloud.google.com/go/vision/v2",
        sum = "h1:kvR1sHcuPYat1wI3BYY7CEX2xLAcUHPYL6dOzV2Xf4Q=",
        version = "v2.8.1",
    )
    go_repository(
        name = "com_google_cloud_go_vmmigration",
        importpath = "cloud.google.com/go/vmmigration",
        sum = "h1:sbaWK76csqtk0TGPGCiJqZi7tfrU0LnrhUjZHI5YVdc=",
        version = "v1.7.6",
    )
    go_repository(
        name = "com_google_cloud_go_vmwareengine",
        importpath = "cloud.google.com/go/vmwareengine",
        sum = "h1:Mf8abigBstvjfSGq9twhtbMTCONL0Cjds+tGbc2pV0M=",
        version = "v1.1.2",
    )
    go_repository(
        name = "com_google_cloud_go_vpcaccess",
        importpath = "cloud.google.com/go/vpcaccess",
        sum = "h1:wbMTRdZ9P5+3D6oQWWqB/YxDCFR5m5OJ+b+hHwaBBQQ=",
        version = "v1.7.6",
    )
    go_repository(
        name = "com_google_cloud_go_webrisk",
        importpath = "cloud.google.com/go/webrisk",
        sum = "h1:rVhi2WOHcZF72X7spXVTFTmRGeFN4NFeW7/Ku7kgeug=",
        version = "v1.9.6",
    )
    go_repository(
        name = "com_google_cloud_go_websecurityscanner",
        importpath = "cloud.google.com/go/websecurityscanner",
        sum = "h1:YAwNB/HjKOVAy9D7W8Bkv8OQ9G2lqIqFOuJbyH5Xo4Q=",
        version = "v1.6.6",
    )
    go_repository(
        name = "com_google_cloud_go_workflows",
        importpath = "cloud.google.com/go/workflows",
        sum = "h1:hH511zmS93oE6j64m/eiGWnfgqailh/S8+f3MVNLcE8=",
        version = "v1.12.5",
    )
    go_repository(
        name = "ht_sr_git_sbinet_cmpimg",
        importpath = "git.sr.ht/~sbinet/cmpimg",
        sum = "h1:E0zPRk2muWuCqSKSVZIWsgtU9pjsw3eKHi8VmQeScxo=",
        version = "v0.1.0",
    )
    go_repository(
        name = "ht_sr_git_sbinet_gg",
        importpath = "git.sr.ht/~sbinet/gg",
        sum = "h1:6V43j30HM623V329xA9Ntq+WJrMjDxRjuAB1LFWF5m8=",
        version = "v0.5.0",
    )
    go_repository(
        name = "in_gopkg_check_v1",
        importpath = "gopkg.in/check.v1",
        sum = "h1:yhCVgyC4o1eVCa2tZl7eS0r+SDo693bJlVdllGtEeKM=",
        version = "v0.0.0-20161208181325-20d25e280405",
    )
    go_repository(
        name = "in_gopkg_retry_v1",
        importpath = "gopkg.in/retry.v1",
        sum = "h1:a9CArYczAVv6Qs6VGoLMio99GEs7kY9UzSF9+LD+iGs=",
        version = "v1.0.3",
    )
    go_repository(
        name = "in_gopkg_yaml_v2",
        importpath = "gopkg.in/yaml.v2",
        sum = "h1:D8xgwECY7CYvx+Y2n4sBz93Jn9JRvxdiyyo8CTfuKaY=",
        version = "v2.4.0",
    )
    go_repository(
        name = "in_gopkg_yaml_v3",
        importpath = "gopkg.in/yaml.v3",
        sum = "h1:fxVm/GzAzEWqLHuvctI91KS9hhNmmWOoWu0XTYJS7CA=",
        version = "v3.0.1",
    )
    go_repository(
        name = "io_opencensus_go",
        importpath = "go.opencensus.io",
        sum = "h1:y73uSU6J157QMP2kn2r30vwW1A2W2WFwSCGnAVxeaD0=",
        version = "v0.24.0",
    )
    go_repository(
        name = "io_opentelemetry_go_contrib_instrumentation_google_golang_org_grpc_otelgrpc",
        importpath = "go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc",
        sum = "h1:4Pp6oUg3+e/6M4C0A/3kJ2VYa++dsWVTtGgLVj5xtHg=",
        version = "v0.49.0",
    )
    go_repository(
        name = "io_opentelemetry_go_contrib_instrumentation_net_http_otelhttp",
        importpath = "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp",
        sum = "h1:jq9TW8u3so/bN+JPT166wjOI6/vQPF6Xe7nMNIltagk=",
        version = "v0.49.0",
    )
    go_repository(
        name = "io_opentelemetry_go_otel",
        importpath = "go.opentelemetry.io/otel",
        sum = "h1:0LAOdjNmQeSTzGBzduGe/rU4tZhMwL5rWgtp9Ku5Jfo=",
        version = "v1.24.0",
    )
    go_repository(
        name = "io_opentelemetry_go_otel_exporters_otlp_otlptrace",
        importpath = "go.opentelemetry.io/otel/exporters/otlp/otlptrace",
        sum = "h1:t6wl9SPayj+c7lEIFgm4ooDBZVb01IhLB4InpomhRw8=",
        version = "v1.24.0",
    )
    go_repository(
        name = "io_opentelemetry_go_otel_exporters_otlp_otlptrace_otlptracehttp",
        importpath = "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp",
        sum = "h1:Xw8U6u2f8DK2XAkGRFV7BBLENgnTGX9i4rQRxJf+/vs=",
        version = "v1.24.0",
    )
    go_repository(
        name = "io_opentelemetry_go_otel_metric",
        importpath = "go.opentelemetry.io/otel/metric",
        sum = "h1:6EhoGWWK28x1fbpA4tYTOWBkPefTDQnb8WSGXlc88kI=",
        version = "v1.24.0",
    )
    go_repository(
        name = "io_opentelemetry_go_otel_sdk",
        importpath = "go.opentelemetry.io/otel/sdk",
        sum = "h1:YMPPDNymmQN3ZgczicBY3B6sf9n62Dlj9pWD3ucgoDw=",
        version = "v1.24.0",
    )
    go_repository(
        name = "io_opentelemetry_go_otel_trace",
        importpath = "go.opentelemetry.io/otel/trace",
        sum = "h1:CsKnnL4dUAr/0llH9FKuc698G04IrpWV0MQA/Y1YELI=",
        version = "v1.24.0",
    )
    go_repository(
        name = "io_opentelemetry_go_proto_otlp",
        importpath = "go.opentelemetry.io/proto/otlp",
        sum = "h1:2Di21piLrCqJ3U3eXGCTPHE9R8Nh+0uglSnOyxikMeI=",
        version = "v1.1.0",
    )
    go_repository(
        name = "io_rsc_pdf",
        importpath = "rsc.io/pdf",
        sum = "h1:k1MczvYDUvJBe93bYd7wrZLLUEcLZAuF824/I4e5Xr4=",
        version = "v0.1.1",
    )
    go_repository(
        name = "org_gioui",
        importpath = "gioui.org",
        sum = "h1:RbzDn1h/pCVf/q44ImQSa/J3MIFpY3OWphzT/Tyei+w=",
        version = "v0.2.0",
    )
    go_repository(
        name = "org_gioui_cpu",
        importpath = "gioui.org/cpu",
        sum = "h1:tNJdnP5CgM39PRc+KWmBRRYX/zJ+rd5XaYxY5d5veqA=",
        version = "v0.0.0-20220412190645-f1e9e8c3b1f7",
    )
    go_repository(
        name = "org_gioui_shader",
        importpath = "gioui.org/shader",
        sum = "h1:cvZmU+eODFR2545X+/8XucgZdTtEjR3QWW6W65b0q5Y=",
        version = "v1.0.6",
    )
    go_repository(
        name = "org_gioui_x",
        importpath = "gioui.org/x",
        sum = "h1:/MbdjKH19F16auv19UiQxli2n6BYPw7eyh9XBOTgmEw=",
        version = "v0.2.0",
    )
    go_repository(
        name = "org_golang_google_api",
        build_directives = [
            # See https://github.com/bazelbuild/rules_go/issues/1877#issuecomment-1874616324
            "gazelle:resolve go google.golang.org/genproto/googleapis/api/annotations @org_golang_google_genproto//googleapis/api/annotations",  # keep
            "gazelle:resolve go google.golang.org/genproto/googleapis/api @org_golang_google_genproto//googleapis/api",  # keep
        ],
        importpath = "google.golang.org/api",
        sum = "h1:w174hnBPqut76FzW5Qaupt7zY8Kql6fiVjgys4f58sU=",
        version = "v0.171.0",
    )
    go_repository(
        name = "org_golang_google_appengine",
        importpath = "google.golang.org/appengine",
        sum = "h1:IhEN5q69dyKagZPYMSdIjS2HqprW324FRQZJcGqPAsM=",
        version = "v1.6.8",
    )
    go_repository(
        name = "org_golang_google_genproto",
        importpath = "google.golang.org/genproto",
        sum = "h1:ePqxpG3LVx+feAUOx8YmR5T7rc0rdzK8DyxM8cQ9zq0=",
        version = "v0.0.0-20240325203815-454cdb8f5daa",
    )
    go_repository(
        name = "org_golang_google_genproto_googleapis_api",
        build_directives = [
            # See https://github.com/bazelbuild/rules_go/issues/1877#issuecomment-1874616324
            "gazelle:resolve go google.golang.org/genproto/googleapis/api/annotations @org_golang_google_genproto//googleapis/api/annotations",  # keep
            "gazelle:resolve go google.golang.org/genproto/googleapis/api @org_golang_google_genproto//googleapis/api",  # keep
        ],
        importpath = "google.golang.org/genproto/googleapis/api",
        sum = "h1:Jt1XW5PaLXF1/ePZrznsh/aAUvI7Adfc3LY1dAKlzRs=",
        version = "v0.0.0-20240325203815-454cdb8f5daa",
    )
    go_repository(
        name = "org_golang_google_genproto_googleapis_bytestream",
        importpath = "google.golang.org/genproto/googleapis/bytestream",
        sum = "h1:4z0DVWmDWWZ4OeQHLrb6lLBE3uCgSLs9DDA5Zb36DFg=",
        version = "v0.0.0-20240314234333-6e1732d8331c",
    )
    go_repository(
        name = "org_golang_google_genproto_googleapis_rpc",
        importpath = "google.golang.org/genproto/googleapis/rpc",
        sum = "h1:RBgMaUMP+6soRkik4VoN8ojR2nex2TqZwjSSogic+eo=",
        version = "v0.0.0-20240325203815-454cdb8f5daa",
    )
    go_repository(
        name = "org_golang_google_grpc",
        build_file_proto_mode = "disable_global",  # See https://github.com/bazelbuild/rules_go/issues/2186#issuecomment-523028281
        importpath = "google.golang.org/grpc",
        sum = "h1:B4n+nfKzOICUXMgyrNd19h/I9oH0L1pizfk1d4zSgTk=",
        version = "v1.62.1",
    )
    go_repository(
        name = "org_golang_google_protobuf",
        importpath = "google.golang.org/protobuf",
        sum = "h1:uNO2rsAINq/JlFpSdYEKIZ0uKD/R9cpdv0T+yoGwGmI=",
        version = "v1.33.0",
    )
    go_repository(
        name = "org_golang_x_crypto",
        importpath = "golang.org/x/crypto",
        sum = "h1:X31++rzVUdKhX5sWmSOFZxx8UW/ldWx55cbf08iNAMA=",
        version = "v0.21.0",
    )
    go_repository(
        name = "org_golang_x_exp",
        importpath = "golang.org/x/exp",
        sum = "h1:aAcj0Da7eBAtrTp03QXWvm88pSyOt+UgdZw2BFZ+lEw=",
        version = "v0.0.0-20240325151524-a685a6edb6d8",
    )
    go_repository(
        name = "org_golang_x_exp_shiny",
        importpath = "golang.org/x/exp/shiny",
        sum = "h1:sgkbz1SFTsoQIvzTIw45hccUcGocu00QM3qucBYV8b0=",
        version = "v0.0.0-20230801115018-d63ba01acd4b",
    )
    go_repository(
        name = "org_golang_x_image",
        importpath = "golang.org/x/image",
        sum = "h1:kOELfmgrmJlw4Cdb7g/QGuB3CvDrXbqEIww/pNtNBm8=",
        version = "v0.15.0",
    )
    go_repository(
        name = "org_golang_x_lint",
        importpath = "golang.org/x/lint",
        sum = "h1:XQyxROzUlZH+WIQwySDgnISgOivlhjIEwaQaJEJrrN0=",
        version = "v0.0.0-20190313153728-d0100b6bd8b3",
    )
    go_repository(
        name = "org_golang_x_mod",
        importpath = "golang.org/x/mod",
        sum = "h1:QX4fJ0Rr5cPQCF7O9lh9Se4pmwfwskqZfq5moyldzic=",
        version = "v0.16.0",
    )
    go_repository(
        name = "org_golang_x_net",
        importpath = "golang.org/x/net",
        sum = "h1:7EYJ93RZ9vYSZAIb2x3lnuvqO5zneoD6IvWjuhfxjTs=",
        version = "v0.23.0",
    )
    go_repository(
        name = "org_golang_x_oauth2",
        importpath = "golang.org/x/oauth2",
        sum = "h1:09qnuIAgzdx1XplqJvW6CQqMCtGZykZWcXzPMPUusvI=",
        version = "v0.18.0",
    )
    go_repository(
        name = "org_golang_x_sync",
        importpath = "golang.org/x/sync",
        sum = "h1:5BMeUDZ7vkXGfEr1x9B4bRcTH4lpkTkpdh0T/J+qjbQ=",
        version = "v0.6.0",
    )
    go_repository(
        name = "org_golang_x_sys",
        importpath = "golang.org/x/sys",
        sum = "h1:DBdB3niSjOA/O0blCZBqDefyWNYveAYMNF1Wum0DYQ4=",
        version = "v0.18.0",
    )
    go_repository(
        name = "org_golang_x_telemetry",
        importpath = "golang.org/x/telemetry",
        sum = "h1:IRJeR9r1pYWsHKTRe/IInb7lYvbBVIqOgsX/u0mbOWY=",
        version = "v0.0.0-20240228155512-f48c80bd79b2",
    )
    go_repository(
        name = "org_golang_x_term",
        importpath = "golang.org/x/term",
        sum = "h1:FcHjZXDMxI8mM3nwhX9HlKop4C0YQvCVCdwYl2wOtE8=",
        version = "v0.18.0",
    )
    go_repository(
        name = "org_golang_x_text",
        importpath = "golang.org/x/text",
        sum = "h1:ScX5w1eTa3QqT8oi6+ziP7dTV1S2+ALU0bI+0zXKWiQ=",
        version = "v0.14.0",
    )
    go_repository(
        name = "org_golang_x_time",
        importpath = "golang.org/x/time",
        sum = "h1:o7cqy6amK/52YcAKIPlM3a+Fpj35zvRj2TP+e1xFSfk=",
        version = "v0.5.0",
    )
    go_repository(
        name = "org_golang_x_tools",
        importpath = "golang.org/x/tools",
        sum = "h1:tfGCXNR1OsFG+sVdLAitlpjAvD/I6dHDKnYrpEZUHkw=",
        version = "v0.19.0",
    )
    go_repository(
        name = "org_golang_x_xerrors",
        importpath = "golang.org/x/xerrors",
        sum = "h1:+cNy6SZtPcJQH3LJVLOSmiC7MMxXNOb3PU/VUEz+EhU=",
        version = "v0.0.0-20231012003039-104605ab7028",
    )
    go_repository(
        name = "org_gonum_v1_gonum",
        importpath = "gonum.org/v1/gonum",
        sum = "h1:2lYxjRbTYyxkJxlhC+LvJIx3SsANPdRybu1tGj9/OrQ=",
        version = "v0.15.0",
    )
    go_repository(
        name = "org_gonum_v1_plot",
        importpath = "gonum.org/v1/plot",
        sum = "h1:+LBDVFYwFe4LHhdP8coW6296MBEY4nQ+Y4vuUpJopcE=",
        version = "v0.14.0",
    )
    go_repository(
        name = "org_mongodb_go_mongo_driver",
        importpath = "go.mongodb.org/mongo-driver",
        sum = "h1:YIc7HTYsKndGK4RFzJ3covLz1byri52x0IoMB0Pt/vk=",
        version = "v1.13.1",
    )
    go_repository(
        name = "tech_einride_go_aip",
        importpath = "go.einride.tech/aip",
        sum = "h1:XfV+NQX6L7EOYK11yoHHFtndeaWh3KbD9/cN/6iWEt8=",
        version = "v0.66.0",
    )
    go_repository(
        name = "tools_gotest_v3",
        importpath = "gotest.tools/v3",
        sum = "h1:EENdUnS3pdur5nybKYIh2Vfgc8IUNBjxDPSjtiJcOzU=",
        version = "v3.5.1",
    )
