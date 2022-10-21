
0. build disc-pdll tool: python scripts/python/tao_build.py /opt/venv_disc/  --cpu_only -s build_tao_compiler

1. Run exampel 1: ./tf_community/bazel-bin/tensorflow/compiler/mlir/disc/disc-pdll --pdl-input ./tao_compiler/mlir/disc/tools/disc-pdll/test.pdll --payload-input ./tao_compiler/mlir/disc/tools/disc-pdll/payload.mlir

2. Run exampel 2: ./tf_community/bazel-bin/tensorflow/compiler/mlir/disc/disc-pdll --pdl-input ./tao_compiler/mlir/disc/tools/disc-pdll/test_2.pdll --payload-input ./tao_compiler/mlir/disc/tools/disc-pdll/payload_2.mlir

