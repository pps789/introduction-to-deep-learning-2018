files="BatchNormalization.ipynb
ConvolutionalNetworks.ipynb
Dropout.ipynb
FullyConnectedNets.ipynb
TensorFlow.ipynb"

for file in $files
do
    if [ ! -f $file ]; then
        echo "Required notebook $file not found."
        exit 0
    fi
done

rm -f hw3_results.zip
zip -r hw3_results.zip . -x "*.git*" "*cs231n/datasets*" "*.ipynb_checkpoints*" "*collectSubmission.sh" "*requirements.txt" ".env/*" "*.pyc" "*cs231n/build/*"
