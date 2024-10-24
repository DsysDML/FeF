#!/bin/bash

# Pressing CTRL-C will stop the whole execution of the script
trap ctrl_c INT; 
function ctrl_c() { exit 5; }

# variables

CMD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)
CMD_NAME=$(basename "${BASH_SOURCE[0]}")
MODEL_STRING=""
TRAIN_OPTIONS=()
MODEL_ID=false
HELP=false

function list_rbm_models() {
    local rbm_dir="${RBMHOME}/src/RBMs"
    echo "Arguments:"
    echo -en "\t-m, --model <model_identifier>\n"
    echo -en "\n\tSpecify one of the following model identifiers:\n"
    for file in "${rbm_dir}"/*RBM.py; do
        local file_name=$(basename "$file")
        local name_without_ext="${file_name%RBM.py}"
        echo -en "\n\t$name_without_ext"
    done
    echo -en "\n\n"
}

while [ -n "${1}" ]; do
    case ${1} in
        -m|--model)
            shift
            MODEL_STRING=${1}
            MODEL_ID=true
            ;;
        -h|--help)
            HELP=true
            ;;
        *)
            TRAIN_OPTIONS+=( $1 )
            shift
            continue
            ;;
        --)
            shift
            break
            ;;
    esac
    shift
done

if $HELP; then
    if $MODEL_ID; then
        TRAIN_OPTIONS+=( "-h" )
    else
        list_rbm_models
        exit 1
    fi
fi

python3 "${RBMHOME}/src/train/train${MODEL_STRING}RBM.py" ${TRAIN_OPTIONS[@]}