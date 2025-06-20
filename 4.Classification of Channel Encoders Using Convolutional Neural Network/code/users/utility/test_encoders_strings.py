import numpy as np
import bchlib
import torch
import torch.nn as nn
import torch.nn.functional as F


# Function to convert text to binary
def text_to_binary(text):
    binary_data = []
    for char in text:
        binary_char = format(ord(char), '08b')  # Convert character to 8-bit binary
        binary_data.extend(map(int, binary_char))
    return np.array(binary_data)


# Simple block encoder (parity bit)
def block_encode(data, block_size=8):
    num_blocks = len(data) // block_size
    encoded_data = []
    for i in range(num_blocks):
        block = data[i * block_size: (i + 1) * block_size]
        encoded_data.append(np.sum(block) % 2)  # Parity bit
    return np.array(encoded_data)


# Convolutional Encoder
class ConvolutionalEncoder(nn.Module):
    def __init__(self):
        super(ConvolutionalEncoder, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = torch.unsqueeze(x, 0)  # Add batch dimension
        x = torch.unsqueeze(x, 0)  # Add channel dimension
        x = F.pad(x, (1, 1))  # Pad input sequence
        x = self.conv(x)
        return torch.sigmoid(x).squeeze()


# BCH Encoder
def bch_encode(data, blkLen):
    # bch_polynomial = 8219
    # bch_bits = 8
    # bch = bchlib.BCH(bch_polynomial, bch_bits)
    # encoded_data = bch.encode(data.tobytes())
    # return np.frombuffer(encoded_data, dtype=np.uint8)
    from sklearn.feature_extraction.text import CountVectorizer
    import random
    # Example text data

    # Create CountVectorizer object
    vectorizer = CountVectorizer()

    # Fit and transform the data
    X = vectorizer.fit_transform([data])
    blk = []
    # Get the feature names (vocabulary)
    feature_names = vectorizer.get_feature_names_out()

    # Print the feature names and the encoded data
    print("Feature Names:", feature_names)
    print("Encoded Data:")
    c = blkLen - len(X.toarray())
    for i in range(blkLen):
        blk.append(random.uniform(0, 1))

    # return X.toarray()
    return blk


# Polar Encoder
def polar_encode(data):
    N = len(data)
    x = np.zeros(N, dtype=int)
    for i in range(N):
        if i == 0:
            x[i] = data[i]
        else:
            x[i] = x[i - 1] ^ data[i]
    return x


def start_process(text):
    # Convert text to binary
    binary_data = text_to_binary(text)

    # Encode using block encoder
    block_encoded_data = block_encode(binary_data)
    encoder_size = 16  # len(block_encoded_data)

    # Encode using convolutional encoder
    conv_encoder = ConvolutionalEncoder()
    conv_encoded_data = conv_encoder(torch.tensor(binary_data, dtype=torch.float32))
    conv_Data = conv_encoded_data.detach().numpy()
    conv_Data = conv_Data[:encoder_size]
    # Encode using BCH encoder
    bch_encoded_data = bch_encode(text, encoder_size)

    # Encode using polar encoder
    polar_encoded_data = polar_encode(binary_data)
    polar_encoded_data = polar_encoded_data[:encoder_size]
    # print("Original Text:", text)
    # print("Binary Data:", binary_data)
    print("Block Encoded Data:", block_encoded_data)
    print("Convolutional Encoded Data:", conv_Data)
    print("BCH Encoded Data:", bch_encoded_data)
    print("Polar Encoded Data:", polar_encoded_data)
    rslt = {'blockencoder': block_encoded_data,
            'convolutionalencoder': conv_Data,
            'bch': bch_encoded_data,
            'polar': polar_encoded_data
            }
    return rslt
