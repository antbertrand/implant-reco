# Install additional modules
pip install --upgrade pip
pip install awscli

# S3 credentials
export AWS_ACCESS_KEY_ID=AKIAIE2OPCN2K47HTOSQ
export AWS_SECRET_ACCESS_KEY=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

# # If we are calling from Paperspace, sync from S3
# if [ ! -f /local-docker-host ]; then
#     STOOOOOOOP
#     echo "**********************************************************************"
#     echo "** Initializing S3 dataset /!\ THIS DOESN'T WORK AS EXPECTED, FIXME **"
#     echo "**********************************************************************"
#     mkdir -p /storage/datasets/
#     aws s3 sync s3://cardamin-paperspace/datasets/cardamin-L2 /storage/datasets/cardamin-L2 --delete
# fi

. ./run.sh
