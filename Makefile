BAZEL_REQUIRED_VERSION=0.26.

BAZEL_PATH=$(shell which bazel)

bazelcheck:
ifeq (,$(BAZEL_PATH))
ifeq (,$(findstring $(BAZEL_REQUIRED_VERSION),$(shell bazel version)))
ifeq (,$(BYPASS_BAZEL_CHECK))
	$(error "Bazel version $(BAZEL_REQUIRED_VERSION) is required.")
endif
endif
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	PYPI_PLATFORM=manylinux1_x86_64
endif
ifeq ($(UNAME_S),Darwin)
	PYPI_PLATFORM=macosx_10_11_x86_64
endif

.PHONY: bazelcheck

.bazelrc:
	TF_NEED_CUDA=0 ./configure.sh

test: .bazelrc bazelcheck
	bazel test ... --test_output=all

fmt:
	cd tf_seal && find . -iname *.h -o -iname *.cc | xargs clang-format -i -style=google

lint:
	cd tf_seal && find . -iname *.h -o -iname *.cc | xargs cpplint --filter=-legal/copyright

build: .bazelrc bazelcheck
	bazel build build_pip_pkg
	PYPI_PLATFORM=$(PYPI_PLATFORM) ./bazel-bin/build_pip_pkg `pwd`/artifacts

clean: bazelcheck
	rm -f .bazelrc || true
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

.PHONY: build test fmt lint clean tensorflow

# ###############################################
# Version Derivation
#
# Rules and variable definitions used to derive the current version of the
# source code. This information is also used for deriving the type of release
# to perform if `make push` is invoked.
# ###############################################
VERSION=$(shell [ -d .git ] && git describe --tags --abbrev=0 2> /dev/null | sed 's/^v//')
EXACT_TAG=$(shell [ -d .git ] && git describe --exact-match --tags HEAD 2> /dev/null | sed 's/^v//')
ifeq (,$(VERSION))
    VERSION=dev
endif
NOT_RC=$(shell git tag --points-at HEAD | grep -v -e -rc)

ifeq (,$(ARTIFACT_LOCATION))
	ARTIFACT_LOCATION=dist
endif

ifeq ($(EXACT_TAG),)
    PUSHTYPE=master
else
    ifeq ($(NOT_RC),)
	PUSHTYPE=release-candidate
    else
	PUSHTYPE=release
    endif
endif

releasecheck:
ifneq (yes,$(RELEASE_CONFIRM))
	$(error "Set RELEASE_CONFIRM=yes to really build and push release artifacts")
endif

.PHONY: releasecheck

# ###############################################
# Targets for building pip packages for pypi
# ##############################################

pypi-version-check:
ifeq (,$(shell grep -e $(VERSION) setup.py))
	$(error "Version specified in setup.py does not match $(VERSION)")
endif

twine:
	pip install --upgrade setuptools wheel twine

pypi-build: twine
	$(MAKE) build

.PHONY: pypi-build pypi-version-check

# ###############################################
# Targets for publishing to pypi
#
# These targets requires a PYPI_USERNAME, PYPI_PASSWORD, and PYPI_PLATFORM
# environment variables to be set to be executed properly.
# ##############################################

pypi-credentials-check:
ifeq (,$(PYPI_USERNAME))
ifeq (,$(PYPI_PASSWORD))
	$(error "Missing PYPI_USERNAME and PYPI_PASSWORD environment variables")
endif
endif

pypi-push-master: pypi-credentials-check

pypi-push-release-candidate: pypi-version-check twine releasecheck pypi-credentials-check
	@echo "Attempting to upload to pypi"
	twine upload -u="$(PYPI_USERNAME)" -p="$(PYPI_PASSWORD)" $(ARTIFACT_LOCATION)/*

pypi-push-release: pypi-version-check pypi-push-release-candidate

pypi-push: pypi-push-$(PUSHTYPE)

.PHONY: pypi-push pypi-push-release pypi-push-release-candidate pypi-push-master pypi-credentials-check twine

# ###############################################
# Pushing Artifacts for a Release
#
# The following are meta-rules for building and pushing various different
# release artifacts to their intended destinations.
# ###############################################

push:
	@echo "Attempting to build and push $(VERSION) with push type $(PUSHTYPE) - $(EXACT_TAG)"
	make pypi-push
	@echo "Done building and pushing artifacts for $(VERSION)"

.PHONY: push
