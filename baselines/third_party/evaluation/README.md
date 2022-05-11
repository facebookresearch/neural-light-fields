# lpips-tensorflow
Tensorflow port for the [PyTorch](https://github.com/richzhang/PerceptualSimilarity) implementation of the [Learned Perceptual Image Patch Similarity (LPIPS)](http://richzhang.github.io/PerceptualSimilarity/) metric.
This is done by exporting the model from PyTorch to ONNX and then to TensorFlow.

## Getting started
### Installation
- Clone this repo.
```bash
git clone https://github.com/alexlee-gk/lpips-tensorflow.git
cd lpips-tensorflow
```
- Install TensorFlow and dependencies from http://tensorflow.org/
- Install other dependencies.
```bash
pip install -r requirements.txt
```

### Using the LPIPS metric
The `lpips` TensorFlow function works with individual images or batches of images.
It also works with images of any spatial dimensions (but the dimensions should be at least the size of the network's receptive field).
This example computes the LPIPS distance between batches of images.
```python
import numpy as np
import tensorflow as tf
import lpips_tf

batch_size = 32
image_shape = (batch_size, 64, 64, 3)
image0 = np.random.random(image_shape)
image1 = np.random.random(image_shape)
image0_ph = tf.placeholder(tf.float32)
image1_ph = tf.placeholder(tf.float32)

distance_t = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex')

with tf.Session() as session:
    distance = session.run(distance_t, feed_dict={image0_ph: image0, image1_ph: image1})
```

## Exporting additional models
### Export PyTorch model to TensorFlow through ONNX
- Clone the PerceptualSimilarity submodule and add it to the PYTHONPATH.
```bash
git submodule update --init --recursive
export PYTHONPATH=PerceptualSimilarity:$PYTHONPATH
```
- Install more dependencies.
```bash
pip install -r requirements-dev.txt
```
- Export the model to ONNX *.onnx and TensorFlow *.pb files in the `models` directory.
```bash
python export_to_tensorflow.py --model net-lin --net alex
```

### Known issues
- The SqueezeNet model cannot be exported since ONNX cannot export one of the operators.
