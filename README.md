# Deep Compression
This is a Pytorch implementation to understand the concept of deep compression for neural networks, implementation of 'Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding' by Song Han, Huizi Mao, William J. Dally

This implementation implements three core methods in the paper - Deep Compression
- Network Pruning - Less Number of Weights
- Weight Sharing - Reduce Storage for Each Remaining Weight
- Huffman Encoding - Entropy of the Total Remaining Weight

# Notes

- Can Prune Convolutional Layers by 3x and Fully Connected Layers by 10x
- Train Connectivity --> Prune Connections --> Train Weights
- Can Prune RNN and LSTM, without hurting the score
