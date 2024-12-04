#!/bin/bash

cd /opt/NeMo # in order to execute code inside the container and not your code in your lustre dir

if [ -z "${RECIPE_FILE}" ]; then
    echo "RECIPE_FILE is not set."
    exit 1
else
    bash ${RECIPE_FILE}
fi
