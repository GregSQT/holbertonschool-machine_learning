#!/usr/bin/env python3

# Import the densenet121 function from the respective module
densenet121 = __import__('7-densenet121').densenet121

if __name__ == '__main__':
    # Build the DenseNet-121 model with growth rate 32 and compression 0.5
    model = densenet121(32, 0.5)

    # Print the summary of the model architecture
    model.summary()
    