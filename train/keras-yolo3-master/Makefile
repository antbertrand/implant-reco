# This is the preliminary Makefile work for our projects.
# A BIG WARNING FOR EVERYBODY: PUT TABS IN THIS FILE AND NOT SPACES!!!!!!

# This is from https://stackoverflow.com/questions/10858261/abort-makefile-if-variable-not-sets
check_defined = \
	$(strip $(foreach 1,$1, \
		$(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
	$(if $(value $1),, \
		$(error Undefined $1$(if $2, ($2))$(if $(value @), \
				required by target '$@')))

# We define our normal release date tag here.
# If you wanna increment it, just use:
#   export DEPLOY_INC=a
# We use eval here to avoid re-calculating the random string at each call (oups)
$(eval TODAY_TAG:=v$(shell date "+%Y-%m-%d-%H%M%S")-$(shell git branch | awk '/\*/ { print $$2; }'))

# Local dir info
HERE=$(shell /bin/pwd)

local: check
	docker run --rm -ti -w /paperspace \
		-v `pwd`:/paperspace \
		-v /tmp/`pwd`/artifacts:/artifacts \
		-v /tmp/:/local-docker-host \
		-v "${DATASET_DIR}":/storage/ \
		-p 8888:8888 \
		ufoym/deepo:cpu \
		/paperspace/run.sh
.PHONY:local


debug: check
	docker run --rm -ti -w /paperspace \
		-v `pwd`:/paperspace \
		-v /tmp/`pwd`/artifacts:/artifacts \
		-v /tmp/:/local-docker-host \
		-v "${DATASET_DIR}":/storage/ \
		-p 8888:8888 \
		ufoym/deepo:cpu \
		/bin/bash
.PHONY:debug

check:
	@:$(call check_defined, DATASET_DIR, Dataset directory, probably pointer to your Dropbox)
.PHONY:check

paperspace: export DEPLOY_ENV:=develop
paperspace:
#	docker-compose -f ./docker-compose.yml -f ./docker-develop.yml build
	@echo "Starting your paperspace stack."
	paperspace jobs create --container ufoym/deepo --machineType V100 --command "/paperspace/run-paperspace.sh" --ports 8888:8888
.PHONY:develop
