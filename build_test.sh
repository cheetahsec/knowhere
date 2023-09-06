#!/usr/bin/env bash

cd /go/src/github.com/milvus-io/knowhere/build
conan install .. --build=missing -o with_ut=True -s compiler.libcxx=libstdc++11 -s build_type=Release
conan build ..

export VERBOSE=1

pip3 uninstall pyknowhere -y

cd /go/src/github.com/milvus-io/knowhere/python
python3 setup.py bdist_wheel
pip3 install dist/pyknowhere-0.0.0-cp38-cp38-linux_x86_64.whl

cd /go/src/github.com/milvus-io/knowhere/tests/python
python3 test_tlsh.py
