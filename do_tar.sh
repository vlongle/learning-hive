#!/bin/bash

LOCAL_FOLDER="cifar_no_updates_contrastive_results"
TARBALL_NAME="cifar_no_updates_contrastive_results.tar.gz"

tar cf - "${LOCAL_FOLDER}" -P | pv -s $(du -sb "${LOCAL_FOLDER}" | awk '{print $1}') | gzip > "${TARBALL_NAME}"