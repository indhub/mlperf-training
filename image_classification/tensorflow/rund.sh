mpirun -np 2 --hostfile /home/ubuntu/efs/hosts -bind-to none -map-by slot -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib bash /home/ubuntu/efs/mlperf-training/image_classification/tensorflow/run.sh

