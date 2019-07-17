.bazelrc:
	TF_NEED_CUDA=0 ./configure.sh

test: .bazelrc
	bazel test ... --test_output=all

fmt:
	cd tf_seal && find . -iname *.h -o -iname *.cc | xargs clang-format -i -style=google

lint:
	cd tf_seal && find . -iname *.h -o -iname *.cc | xargs cpplint --filter=-legal/copyright

clean:
	bazel clean

c++17.tar.gz:
	wget https://github.com/dropoutlabs/tensorflow/archive/c++17.tar.gz

tensorflow-c-17: c++17.tar.gz
	tar -xf c++17.tar.gz

tensorflow: tensorflow-c-17
	cd tensorflow-c-17 && echo -e '\n' | TF_ENABLE_XLA=0 TF_NEED_CUDA=0 TF_SET_ANDROID_WORKSPACE=0 \
	 TF_CONFIGURE_IOS=0 TF_NEED_OPENCL_SYCL=0 TF_DOWNLOAD_CLANG=0 \
	 TF_NEED_ROCM=0 TF_NEED_MPI=0 ./configure
	cd tensorflow-c-17 && bazel build --config=opt --config=c++17 --config=noaws --config=nogcp \
		--config=nohdfs --config=noignite --config=nokafka --config=nonccl \
		//tensorflow/tools/pip_package:build_pip_package
	cd tensorflow-c-17 && ./bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag pkgs

.PHONY: test fmt lint clean tensorflow
