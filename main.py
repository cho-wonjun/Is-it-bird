import os
import shutil
import sys
import cv2
import numpy as np

# amount of data
# not birds
# 2572
# 548
# 552
# birds
# 568
# 112
# 135
np.set_printoptions(threshold=sys.maxsize)

TRAIN_DATASET_PATHS = ['./data/birds/train', './data/not_birds/train']

def load_images(folder_paths):
    """
    returns: (num_images, 128, 128, 3), (num_images, 0 or 1)
    """
    X = []
    y = []

    for folder_path in folder_paths:
        for image in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0

            X.append(img)
            
            if 'not_birds' in folder_path:
                y.append(0)
            else:
                y.append(1)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y

def dense_forward(X, W, b):
    """
    X: input
    W: weight
    b: bias
    returns: linear regression output
    """
    return np.dot(X, W) + b

def dense_backward(d_out, X, W):
    """
    d_out: gradient
    X: input during forward pass
    W: weights
    returns: dX, dW, db
    """
    # dL/dW = X^T * dL/dz
    dW = np.dot(X.T, d_out)
    # dL/db = sum over batch
    db = np.sum(d_out, axis=0, keepdims=True)
    # dL/dX = dL/dz * W^T
    dX = np.dot(d_out, W.T)
    
    return dX, dW, db

def conv_forward(X, kernels, stride=1):
    """
    X: (H, W, C)
    kernels: (num_filters, kH, kW, kC)
    stride: step size
    returns: (num_filters, out_h, out_w)
    """
    H, W, C = X.shape
    num_filters, kH, kW, kC = kernels.shape
    
    assert C == kC # input channel must match
    
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1
    
    output = np.zeros((num_filters, out_h, out_w))
    
    for f in range(num_filters):
        kernel = kernels[f]
        for i in range(out_h):
            for j in range(out_w):
                patch = X[i:i+kH, j:j+kW, :]
                output[f, i, j] = np.sum(patch * kernel)
    
    return output

def conv_backward(d_out, X, W, stride=1):
    """
    d_out: gradient (F, out_h, out_w)
    X: (H, W, C)
    W: kernels (num_filters, kH, kW, kC)
    stride: step size
    returns: dX, dW, db
    """
    F, out_h, out_w = d_out.shape
    H, W_in, C = X.shape
    num_filters, kH, kW, kC = W.shape
    
    dX = np.zeros_like(X)
    dW = np.zeros_like(W)
    db = np.zeros((F, ))
    
    for f in range(F):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                h_end = h_start + kH
                w_start = j * stride
                w_end = w_start + kW
                
                patch = X[h_start:h_end, w_start:w_end, :]
                
                # Gradient respect to weight
                dW[f] += patch * d_out[f, i, j]
                db[f] += d_out[f, i, j]
                # Gradient respect to input
                dX[h_start:h_end, w_start:w_end, :] += W[f] * d_out[f, i, j]
    
    return dX, dW, db

def max_pool(feature_maps, size=2, stride=2):
    """
    feature_maps: (num_filters, H, W)
    size: pooling window size
    stride: step size
    returns: (num_filters, out_h, out_w)
    """
    num_filters, H, W = feature_maps.shape
    
    out_h = (H - size) // stride + 1
    out_w = (W - size) // stride + 1
    
    output = np.zeros((num_filters, out_h, out_w))
    
    for f in range(num_filters):
        for i in range(out_h):
            for j in range(out_w):
                patch = feature_maps[f, i*stride:i*stride+size, j*stride:j*stride+size]
                output[f, i, j] = np.max(patch)
    
    return output

def max_pool_backward(d_out, feature_maps, size=2, stride=2):
    """
    d_out: gradient
    feature_maps: input during forward pass
    """
    num_filters, H, W = feature_maps.shape
    out_h, out_w = d_out.shape[1], d_out.shape[2]
    
    dX = np.zeros_like(feature_maps)
    
    for f in range(num_filters):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                h_end = h_start + size
                w_start = j * stride
                w_end = w_start + size
                
                patch = feature_maps[f, h_start:h_end, w_start:w_end]
                max_val = np.max(patch)
                
                mask = (patch == max_val)
                # += to accumulate all gradient from window overlapping stride < size
                dX[f, h_start:h_end, w_start:w_end] += mask * d_out[f, i, j]
    
    return dX

def relu(x):
    return np.maximum(0, x)

def relu_backward(d_out, X):
    """
    d_out: gradient from next layer
    X: input to relu in foward pass
    """
    dX = d_out.copy()
    dX[X <= 0] = 0
    return dX

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(y_true, y_pred, epsilon=1e-8):
    # Avoid log(0) and log(1) by adding epsilon
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def binary_cross_entropy_derivative(y_true, y_pred):
    # divide by batch size
    return (y_pred - y_true) / y_true.shape[0]


# Initialize weights
rng = np.random.default_rng(123)

# Conv 1 layer: 16 filters, 3x3, in_channels=3
F1, k = 16, 3
C_in = 3
std1 = np.sqrt(2.0 / (k * k * C_in))
filters1 = rng.normal(0, std1, size=(F1, k, k, C_in))

# Conv 2 layer: 32 filters, 3x3, in_channels=F1
F2 = 32
std2 = np.sqrt(2.0 / (k * k * F1))
filters2 = rng.normal(0, std2, size=(F2, k, k, F1))

# Maxpool
# Input 128 -> conv1 out: 126 -> maxpool out: 63
# Input 63 -> conv2 out: 61 -> maxpool out: 30
pool1_HW = 63
pool2_HW = 30
flat_dim = F2 * pool2_HW * pool2_HW

# Dense 1 Layer
neurons = 128
std1_dense = np.sqrt(2.0 / flat_dim)
W1 = rng.normal(0, std1_dense, size=(flat_dim, neurons))
b1 = np.zeros((1, neurons))

# Dense 2 Layer
std2_dense = np.sqrt(2.0 / neurons)
W2 = rng.normal(0, std2_dense, size=(neurons, 1))
b2 = np.zeros((1, 1))

# Hyperparameters
learning_rate = 0.01
batch_size = 32
epochs = 5

class LegacyCode:
    # legacy code that were extended later to add scalability
    def conv_single_filter(img, kernel, stride=1):
        H, W = img.shape
        KH, KW = kernel.shape
        
        out_h = (H - KH) // stride + 1
        out_w = (W - KW) // stride + 1
        
        output = np.zeros((out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                patch = img[i:i+KH, j:j+KW]
                output[i, j] = np.sum(patch * kernel)

        return output

class FileUtility:
    # script dumps for data organization - not clean code nor scaled
    def distribute_data(p):
        source_dir = f'./not_birds/flower_photos/{p}'
        
        train_dest = './not_birds/train'
        dev_dest = './not_birds/dev'
        test_dest = './not_birds/test'
        
        files = os.listdir(source_dir)

        train_num = int(len(files) * 0.7)
        dev_num = int((len(files) - train_num) / 2)
        test_num = len(files) - train_num - dev_num
        
        for i in range(len(files)):
            src = os.path.join(f'./not_birds/flower_photos/{p}', files[i])
            if i >= dev_num + train_num + 1:
                des = os.path.join(test_dest, files[i])
            elif i >= train_num + 1:
                des = os.path.join(dev_dest, files[i])
            else:
                des = os.path.join(train_dest, files[i])
            
            try:
                shutil.move(src, des)
                print(f'Moved: {files[i]}')
                pass
            except Exception as e:
                print(f'Error moving {files[i]}: {e}')
            
                
        print(len(files), train_num, dev_num, test_num)
        print(len(os.listdir('./not_birds/train')))
        print(len(os.listdir('./not_birds/test')))
        print(len(os.listdir('./not_birds/dev')))
        
        print('birds')
        print(len(os.listdir('./birds/train')))
        print(len(os.listdir('./birds/test')))
        print(len(os.listdir('./birds/dev')))

    def move_files_with_auto_rename(src_dir, dest_dir):
        def get_unique_filename(dest_dir, filename):
            name, ext = os.path.splitext(filename)
            candidate = filename
            counter = 1

            while os.path.exists(os.path.join(dest_dir, candidate)):
                if counter == 1:
                    candidate = f"{name} copy{ext}"
                else:
                    candidate = f"{name} copy ({counter}){ext}"
                counter += 1

            return candidate
        
        if not os.path.exists(src_dir):
            print(f"Source directory does not exist: {src_dir}")
            return
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        for filename in os.listdir(src_dir):
            src_path = os.path.join(src_dir, filename)

            if os.path.isfile(src_path):
                new_filename = get_unique_filename(dest_dir, filename)
                dest_path = os.path.join(dest_dir, new_filename)
                shutil.move(src_path, dest_path)
                print(f"Moved: {filename} -> {new_filename}")

    source_directory = "./birds/train/FLAMINGO"
    destination_directory = "./birds/train"
