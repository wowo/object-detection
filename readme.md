    git clone https://github.com/tensorflow/models.git
    cd models/research
    protoc object_detection/protos/*.proto --python_out=.
    export PYTHONPATH=`pwd`/models/research/object_detection:`pwd`/models/research
    export PUSHBULLET_API_KEY=apikey
