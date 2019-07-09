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

.PHONY: test fmt lint clean
