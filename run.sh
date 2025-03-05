#!/bin/bash
# List of n_layer values to try
for n in 2 4 6 8 10; do
    export N_LAYER=$n
    echo "Starting training with n_layer=$n"
    python train.py config/train_tinystories.py &
done

# Wait for all background processes to finish
wait
echo "All training processes finished."