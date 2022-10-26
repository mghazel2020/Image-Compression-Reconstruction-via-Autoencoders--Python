# MNIST Image Compression and Reconstruction using Autoencoders in Python 

<img src="images/Auto-Encoders.png" width="1000"/>
  
  
## 1. Objectives

The objective of this project is to demonstrate how to compress and reconstruct MNIST images using Autoencoders.

## 2.  Autoencoders

Autoencoders are an unsupervised learning technique in which we leverage neural networks for the task of representation learning. Specifically, they have a network architecture, which imposes a bottleneck in the network which forces a compressed knowledge representation of the original input. If the input features were each independent of one another, this compression and subsequent reconstruction would be a very difficult task. However, if some sort of structure exists in the data (i.e. correlations between input features), this structure can be learned and consequently leveraged when forcing the input through the network's bottleneck.

We shall illustrate how to  compress and reconstruct MNIST images using Autoencoders. This can be done because image pixels are highly correlated and there is a significant amount of redundancy, which can be encoded and compressed by the autoencoder while generating reasonably good quality reconstructions.


## 3. Data

We shall illustrate the PCA representation of the  MNIST database of handwritten digits, available from this page, which has a training set of 42,000 examples, and a test set of 18,000 examples. We shall illustrate sample images from this data sets in the next section.

## 4. Development

* Project: MNIST Dataset Image Compression and Reconstruction using Autoencoders:
* The objective of this project is to demonstrate how to compress and reconstruct MNIST images using Autoencoders.

* Author: Mohsen Ghazel (mghazel)
* Date: April 9th, 2021

### 4.1. Part 1: Python imports and global variables:

#### 4.1.1. Standard scientific Python imports:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Python imports and environment setup</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># opencv</span>
<span style="color:#200080; font-weight:bold; ">import</span> cv2
<span style="color:#595979; "># numpy</span>
<span style="color:#200080; font-weight:bold; ">import</span> numpy <span style="color:#200080; font-weight:bold; ">as</span> np
<span style="color:#595979; "># matplotlib</span>
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>pyplot <span style="color:#200080; font-weight:bold; ">as</span> plt

<span style="color:#595979; "># import tansorflow</span>
<span style="color:#595979; ">#----------------------------------</span>
<span style="color:#595979; "># Importing: Tensorflow 2.0 </span>
<span style="color:#595979; "># resulted in the following error:</span>
<span style="color:#595979; ">#---------------------------------</span>
<span style="color:#595979; "># AttributeError: module 'tensorflow' </span>
<span style="color:#595979; "># has no attribute 'placeholder'</span>
<span style="color:#595979; ">#---------------------------------</span>
<span style="color:#595979; "># import tensorflow as tf</span>
<span style="color:#595979; ">#---------------------------------</span>
<span style="color:#595979; "># We need to revert to using </span>
<span style="color:#595979; "># Tensorflow ver 1.x</span>
<span style="color:#595979; ">#---------------------------------</span>
<span style="color:#200080; font-weight:bold; ">import</span> tensorflow<span style="color:#308080; ">.</span>compat<span style="color:#308080; ">.</span>v1 <span style="color:#200080; font-weight:bold; ">as</span> tf
tf<span style="color:#308080; ">.</span>disable_v2_behavior<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span> 
<span style="color:#595979; ">#---------------------------------</span>
<span style="color:#595979; "># import additional functionalities</span>
<span style="color:#200080; font-weight:bold; ">from</span> __future__ <span style="color:#200080; font-weight:bold; ">import</span> print_function<span style="color:#308080; ">,</span> division
<span style="color:#200080; font-weight:bold; ">from</span> builtins <span style="color:#200080; font-weight:bold; ">import</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">,</span> <span style="color:#400000; ">input</span>

<span style="color:#595979; "># import shuffle  from sklearn</span>
<span style="color:#200080; font-weight:bold; ">from</span> sklearn<span style="color:#308080; ">.</span>utils <span style="color:#200080; font-weight:bold; ">import</span> shuffle

<span style="color:#595979; "># import pandas</span>
<span style="color:#200080; font-weight:bold; ">import</span> pandas <span style="color:#200080; font-weight:bold; ">as</span> pd

<span style="color:#595979; "># random number generators values</span>
<span style="color:#595979; "># seed for reproducing the random number generation</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> seed
<span style="color:#595979; "># random integers: I(0,M)</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> randint
<span style="color:#595979; "># random standard unform: U(0,1)</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> random
<span style="color:#595979; "># time</span>
<span style="color:#200080; font-weight:bold; ">import</span> datetime
<span style="color:#595979; "># I/O</span>
<span style="color:#200080; font-weight:bold; ">import</span> os
<span style="color:#595979; "># sys</span>
<span style="color:#200080; font-weight:bold; ">import</span> sys

<span style="color:#595979; "># display figure within the notebook</span>
<span style="color:#44aadd; ">%</span>matplotlib inline

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Test imports and display package versions</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Testing the OpenCV version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"OpenCV : "</span><span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>
<span style="color:#595979; "># Testing the numpy version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Numpy : "</span><span style="color:#308080; ">,</span>np<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>

OpenCV <span style="color:#308080; ">:</span>  <span style="color:#008000; ">3.4</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">8</span>
Numpy <span style="color:#308080; ">:</span>  <span style="color:#008000; ">1.19</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">2</span>
</pre>

#### 4.1.2. Global variables:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># set the random_state seed = 100 for reproducibilty</span>
random_state_seed <span style="color:#308080; ">=</span> <span style="color:#008c00; ">100</span>

<span style="color:#595979; "># the number of visualized images</span>
num_visualized_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">25</span>
</pre>


### 4.2. Part 2: Load MNIST Dataset:

* We use the MINIST dataset, which was downloaded from the following link:

  * Kaggle: Digit Recognizer: https://www.kaggle.com/c/digit-recognizer/data
  * The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.
  * Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.
  * The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.
  * Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero)

#### 4.2.1. Load and normalize the training data set:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># read the training data set</span>
data <span style="color:#308080; ">=</span> pd<span style="color:#308080; ">.</span>read_csv<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'../large_files/train.csv'</span><span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>values<span style="color:#308080; ">.</span>astype<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>float32<span style="color:#308080; ">)</span>
<span style="color:#595979; "># normalize the training data to [0,1]:</span>
x_train <span style="color:#308080; ">=</span> data<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span> <span style="color:#44aadd; ">/</span> <span style="color:#008c00; ">255</span>
<span style="color:#595979; "># format the class type to integer</span>
y_train <span style="color:#308080; ">=</span> data<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">.</span>astype<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>int32<span style="color:#308080; ">)</span>
<span style="color:#595979; "># shuffle the data</span>
x_train<span style="color:#308080; ">,</span> y_train <span style="color:#308080; ">=</span> shuffle<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">,</span> y_train<span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Display a summary of the training data:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># the number of training images</span>
num_train_images <span style="color:#308080; ">=</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Training data:"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"x_train.shape: "</span><span style="color:#308080; ">,</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"y_train.shape: "</span><span style="color:#308080; ">,</span> y_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Number of training images: "</span><span style="color:#308080; ">,</span> num_train_images<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Image size: "</span><span style="color:#308080; ">,</span> x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Classes/labels:"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'The target labels: '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>unique<span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Training data<span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">42000</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">784</span><span style="color:#308080; ">)</span>
y_train<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">42000</span><span style="color:#308080; ">,</span><span style="color:#308080; ">)</span>
Number of training images<span style="color:#308080; ">:</span>  <span style="color:#008c00; ">42000</span>
Image size<span style="color:#308080; ">:</span>  <span style="color:#308080; ">(</span><span style="color:#008c00; ">784</span><span style="color:#308080; ">,</span><span style="color:#308080; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Classes<span style="color:#44aadd; ">/</span>labels<span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The target labels<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span> <span style="color:#008c00; ">1</span> <span style="color:#008c00; ">2</span> <span style="color:#008c00; ">3</span> <span style="color:#008c00; ">4</span> <span style="color:#008c00; ">5</span> <span style="color:#008c00; ">6</span> <span style="color:#008c00; ">7</span> <span style="color:#008c00; ">8</span> <span style="color:#008c00; ">9</span><span style="color:#308080; ">]</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

#### 4.2.2. Visualize some of the training images and their associated targets:

##### 4.2.2.1. First implement a visualization functionality to visualize the number of randomly selected images:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">"""</span>
<span style="color:#595979; "># A utility function to visualize multiple images:</span>
<span style="color:#595979; ">"""</span>
<span style="color:#200080; font-weight:bold; ">def</span> visualize_images_and_labels<span style="color:#308080; ">(</span>num_visualized_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">25</span><span style="color:#308080; ">,</span> dataset_flag <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
  <span style="color:#595979; ">"""To visualize images.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keyword arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- num_visualized_images -- the number of visualized images (deafult 25)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- dataset_flag -- 1: training dataset, 2: test dataset</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Return:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- None</span>
<span style="color:#595979; ">&nbsp;&nbsp;"""</span>
  <span style="color:#595979; ">#--------------------------------------------</span>
  <span style="color:#595979; "># the suplot grid shape:</span>
  <span style="color:#595979; ">#--------------------------------------------</span>
  num_rows <span style="color:#308080; ">=</span> <span style="color:#008c00; ">5</span>
  <span style="color:#595979; "># the number of columns</span>
  num_cols <span style="color:#308080; ">=</span> num_visualized_images <span style="color:#44aadd; ">//</span> num_rows
  <span style="color:#595979; "># setup the subplots axes</span>
  fig<span style="color:#308080; ">,</span> axes <span style="color:#308080; ">=</span> plt<span style="color:#308080; ">.</span>subplots<span style="color:#308080; ">(</span>nrows<span style="color:#308080; ">=</span>num_rows<span style="color:#308080; ">,</span> ncols<span style="color:#308080; ">=</span>num_cols<span style="color:#308080; ">,</span> figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">8</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
  <span style="color:#595979; "># set a seed random number generator for reproducible results</span>
  seed<span style="color:#308080; ">(</span>random_state_seed<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># iterate over the sub-plots</span>
  <span style="color:#200080; font-weight:bold; ">for</span> row <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>num_rows<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
      <span style="color:#200080; font-weight:bold; ">for</span> col <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>num_cols<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># get the next figure axis</span>
        ax <span style="color:#308080; ">=</span> axes<span style="color:#308080; ">[</span>row<span style="color:#308080; ">,</span> col<span style="color:#308080; ">]</span><span style="color:#308080; ">;</span>
        <span style="color:#595979; "># turn-off subplot axis</span>
        ax<span style="color:#308080; ">.</span>set_axis_off<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#595979; "># if the dataset_flag = 1: Training data set</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#308080; ">(</span> dataset_flag <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">1</span> <span style="color:#308080; ">)</span><span style="color:#308080; ">:</span> 
          <span style="color:#595979; "># generate a random image counter</span>
          counter <span style="color:#308080; ">=</span> randint<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>num_train_images<span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the training image</span>
          image <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">,</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the target associated with the image</span>
          label <span style="color:#308080; ">=</span> y_train<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">]</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#595979; "># dataset_flag = 2: Test data set</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span> 
          <span style="color:#595979; "># generate a random image counter</span>
          counter <span style="color:#308080; ">=</span> randint<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>num_test_images<span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the test image</span>
          image <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>x_test<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">,</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the target associated with the image</span>
          label <span style="color:#308080; ">=</span> y_test<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">]</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#595979; "># display the image</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        ax<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>image<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#008c00; ">28</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">28</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> cmap<span style="color:#308080; ">=</span>plt<span style="color:#308080; ">.</span>cm<span style="color:#308080; ">.</span>gray_r<span style="color:#308080; ">,</span> interpolation<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'nearest'</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; "># set the title showing the image label</span>
        ax<span style="color:#308080; ">.</span>set_title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'y ='</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>label<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> size <span style="color:#308080; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#308080; ">)</span>
</pre>

##### 4.2.2.2. Call the function to visualize the randomly selected training images:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># the number of selected training images</span>
num_visualized_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">25</span>
<span style="color:#595979; "># call the function to visualize the training images</span>
visualize_images_and_labels<span style="color:#308080; ">(</span>num_visualized_images<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
</pre>

 <img src="images/sample-train-images.png" width="1000"/>
 
 #### 4.2.3. Examine the number of images for each class of the training and testing subsets:
 
 <pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># create a histogram of the number of images in each class/digit:</span>
<span style="color:#200080; font-weight:bold; ">def</span> plot_bar<span style="color:#308080; ">(</span>y<span style="color:#308080; ">,</span> loc<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">,</span> relative<span style="color:#308080; ">=</span><span style="color:#074726; ">True</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    width <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.35</span>
    <span style="color:#200080; font-weight:bold; ">if</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">:</span>
        n <span style="color:#308080; ">=</span> <span style="color:#44aadd; ">-</span><span style="color:#008000; ">0.5</span>
    <span style="color:#200080; font-weight:bold; ">elif</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'right'</span><span style="color:#308080; ">:</span>
        n <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.5</span>
     
    <span style="color:#595979; "># calculate counts per type and sort, to ensure their order</span>
    unique<span style="color:#308080; ">,</span> counts <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>unique<span style="color:#308080; ">(</span>y<span style="color:#308080; ">,</span> return_counts<span style="color:#308080; ">=</span><span style="color:#074726; ">True</span><span style="color:#308080; ">)</span>
    sorted_index <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>argsort<span style="color:#308080; ">(</span>unique<span style="color:#308080; ">)</span>
    unique <span style="color:#308080; ">=</span> unique<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span>
     
    <span style="color:#200080; font-weight:bold; ">if</span> relative<span style="color:#308080; ">:</span>
        <span style="color:#595979; "># plot as a percentage</span>
        counts <span style="color:#308080; ">=</span> <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">*</span>counts<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span><span style="color:#44aadd; ">/</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y<span style="color:#308080; ">)</span>
        ylabel_text <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'% count'</span>
    <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># plot counts</span>
        counts <span style="color:#308080; ">=</span> counts<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span>
        ylabel_text <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'count'</span>
         
    xtemp <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>arange<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>unique<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>bar<span style="color:#308080; ">(</span>xtemp <span style="color:#44aadd; ">+</span> n<span style="color:#44aadd; ">*</span>width<span style="color:#308080; ">,</span> counts<span style="color:#308080; ">,</span> align<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'center'</span><span style="color:#308080; ">,</span> alpha<span style="color:#308080; ">=</span><span style="color:#008000; ">.7</span><span style="color:#308080; ">,</span> width<span style="color:#308080; ">=</span>width<span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>xticks<span style="color:#308080; ">(</span>xtemp<span style="color:#308080; ">,</span> unique<span style="color:#308080; ">,</span> rotation<span style="color:#308080; ">=</span><span style="color:#008c00; ">45</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'digit'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span>ylabel_text<span style="color:#308080; ">)</span>
 
plt<span style="color:#308080; ">.</span>suptitle<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Frequency of images per digit'</span><span style="color:#308080; ">)</span>
plot_bar<span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">,</span> loc<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>legend<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span>
    <span style="color:#1060b6; ">'train ({0} images)'</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
</pre>
 
 <img src="images/train-images-histogram.png" width="1000"/>
 
### 4.3. Part 3: Implement and train Autoencoder:

#### 4.3.1. Implement the Autoencoder class:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">class</span> Autoencoder<span style="color:#308080; ">:</span>
  <span style="color:#200080; font-weight:bold; ">def</span> <span style="color:#074726; ">__init__</span><span style="color:#308080; ">(</span>self<span style="color:#308080; ">,</span> D<span style="color:#308080; ">,</span> M<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Autoencoder constructor</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    <span style="color:#595979; "># represents a batch of training data</span>
    self<span style="color:#308080; ">.</span>X <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>placeholder<span style="color:#308080; ">(</span>tf<span style="color:#308080; ">.</span>float32<span style="color:#308080; ">,</span> shape<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#074726; ">None</span><span style="color:#308080; ">,</span> D<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

    <span style="color:#595979; "># input -&gt; hidden</span>
    self<span style="color:#308080; ">.</span>W <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>Variable<span style="color:#308080; ">(</span>tf<span style="color:#308080; ">.</span>random_normal<span style="color:#308080; ">(</span>shape<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span>D<span style="color:#308080; ">,</span> M<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">*</span> np<span style="color:#308080; ">.</span>sqrt<span style="color:#308080; ">(</span><span style="color:#008000; ">2.0</span> <span style="color:#44aadd; ">/</span> M<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    self<span style="color:#308080; ">.</span>b <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>Variable<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span>M<span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>astype<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>float32<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

    <span style="color:#595979; "># hidden -&gt; output</span>
    self<span style="color:#308080; ">.</span>V <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>Variable<span style="color:#308080; ">(</span>tf<span style="color:#308080; ">.</span>random_normal<span style="color:#308080; ">(</span>shape<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span>M<span style="color:#308080; ">,</span> D<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">*</span> np<span style="color:#308080; ">.</span>sqrt<span style="color:#308080; ">(</span><span style="color:#008000; ">2.0</span> <span style="color:#44aadd; ">/</span> D<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    self<span style="color:#308080; ">.</span>c <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>Variable<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span>D<span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>astype<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>float32<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

    <span style="color:#595979; "># construct the reconstruction</span>
    self<span style="color:#308080; ">.</span>Z <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>nn<span style="color:#308080; ">.</span>relu<span style="color:#308080; ">(</span>tf<span style="color:#308080; ">.</span>matmul<span style="color:#308080; ">(</span>self<span style="color:#308080; ">.</span>X<span style="color:#308080; ">,</span> self<span style="color:#308080; ">.</span>W<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> self<span style="color:#308080; ">.</span>b<span style="color:#308080; ">)</span>
    logits <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>matmul<span style="color:#308080; ">(</span>self<span style="color:#308080; ">.</span>Z<span style="color:#308080; ">,</span> self<span style="color:#308080; ">.</span>V<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> self<span style="color:#308080; ">.</span>c
    self<span style="color:#308080; ">.</span>X_hat <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>nn<span style="color:#308080; ">.</span>sigmoid<span style="color:#308080; ">(</span>logits<span style="color:#308080; ">)</span>

    <span style="color:#595979; "># compute the cost</span>
    self<span style="color:#308080; ">.</span>cost <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>reduce_sum<span style="color:#308080; ">(</span>
      tf<span style="color:#308080; ">.</span>nn<span style="color:#308080; ">.</span>sigmoid_cross_entropy_with_logits<span style="color:#308080; ">(</span>
        labels<span style="color:#308080; ">=</span>self<span style="color:#308080; ">.</span>X<span style="color:#308080; ">,</span>
        logits<span style="color:#308080; ">=</span>logits
      <span style="color:#308080; ">)</span>
    <span style="color:#308080; ">)</span>

    <span style="color:#595979; "># create the trainer</span>
    self<span style="color:#308080; ">.</span>train_op <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>train<span style="color:#308080; ">.</span>RMSPropOptimizer<span style="color:#308080; ">(</span>learning_rate<span style="color:#308080; ">=</span><span style="color:#008000; ">0.001</span><span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>minimize<span style="color:#308080; ">(</span>self<span style="color:#308080; ">.</span>cost<span style="color:#308080; ">)</span>

    <span style="color:#595979; "># set up session and variables for later</span>
    self<span style="color:#308080; ">.</span>init_op <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>global_variables_initializer<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    self<span style="color:#308080; ">.</span>sess <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>InteractiveSession<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    self<span style="color:#308080; ">.</span>sess<span style="color:#308080; ">.</span>run<span style="color:#308080; ">(</span>self<span style="color:#308080; ">.</span>init_op<span style="color:#308080; ">)</span>

  <span style="color:#200080; font-weight:bold; ">def</span> fit<span style="color:#308080; ">(</span>self<span style="color:#308080; ">,</span> X<span style="color:#308080; ">,</span> epochs<span style="color:#308080; ">=</span><span style="color:#008c00; ">30</span><span style="color:#308080; ">,</span> batch_sz<span style="color:#308080; ">=</span><span style="color:#008c00; ">64</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Fit the model</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    costs <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span><span style="color:#308080; ">]</span>
    n_batches <span style="color:#308080; ">=</span> <span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>X<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">//</span> batch_sz
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"-----------------------------"</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"n_batches:"</span><span style="color:#308080; ">,</span> n_batches<span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">for</span> i <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>epochs<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
      <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"-----------------------------"</span><span style="color:#308080; ">)</span>
      <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"epoch:"</span><span style="color:#308080; ">,</span> i<span style="color:#308080; ">)</span>
      <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"-----------------------------"</span><span style="color:#308080; ">)</span>
      np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>shuffle<span style="color:#308080; ">(</span>X<span style="color:#308080; ">)</span>
      <span style="color:#200080; font-weight:bold; ">for</span> j <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>n_batches<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        batch <span style="color:#308080; ">=</span> X<span style="color:#308080; ">[</span>j<span style="color:#44aadd; ">*</span>batch_sz<span style="color:#308080; ">:</span><span style="color:#308080; ">(</span>j<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#44aadd; ">*</span>batch_sz<span style="color:#308080; ">]</span>
        _<span style="color:#308080; ">,</span> c<span style="color:#308080; ">,</span> <span style="color:#308080; ">=</span> self<span style="color:#308080; ">.</span>sess<span style="color:#308080; ">.</span>run<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>self<span style="color:#308080; ">.</span>train_op<span style="color:#308080; ">,</span> self<span style="color:#308080; ">.</span>cost<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> feed_dict<span style="color:#308080; ">=</span><span style="color:#406080; ">{</span>self<span style="color:#308080; ">.</span>X<span style="color:#308080; ">:</span> batch<span style="color:#406080; ">}</span><span style="color:#308080; ">)</span>
        c <span style="color:#44aadd; ">/</span><span style="color:#308080; ">=</span> batch_sz 
        costs<span style="color:#308080; ">.</span>append<span style="color:#308080; ">(</span>c<span style="color:#308080; ">)</span>
        <span style="color:#595979; "># display the cost for selected epochs</span>
        <span style="color:#200080; font-weight:bold; ">if</span> j <span style="color:#44aadd; ">%</span> <span style="color:#008c00; ">200</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">:</span>
          <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"iter: %d, cost: %.3f"</span> <span style="color:#44aadd; ">%</span> <span style="color:#308080; ">(</span>j<span style="color:#308080; ">,</span> c<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"-----------------------------"</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Training completed successfully!'</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"-----------------------------"</span><span style="color:#308080; ">)</span>
    <span style="color:#595979; ">#----------------------------------------</span>
    <span style="color:#595979; "># display the cost/function as a function </span>
    <span style="color:#595979; "># of the epochs</span>
    <span style="color:#595979; ">#----------------------------------------</span>
    <span style="color:#595979; "># create a figure and set its axis</span>
    fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">8</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">5</span><span style="color:#308080; ">)</span>
    <span style="color:#595979; "># create the figure </span>
    plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>plot<span style="color:#308080; ">(</span>costs<span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Training lossas function of the epoch number"</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Epoch number"</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Loss"</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
        
  <span style="color:#200080; font-weight:bold; ">def</span> predict<span style="color:#308080; ">(</span>self<span style="color:#308080; ">,</span> X<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Generate Autoencoder reconstruction of the input X</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    <span style="color:#200080; font-weight:bold; ">return</span> self<span style="color:#308080; ">.</span>sess<span style="color:#308080; ">.</span>run<span style="color:#308080; ">(</span>self<span style="color:#308080; ">.</span>X_hat<span style="color:#308080; ">,</span> feed_dict<span style="color:#308080; ">=</span><span style="color:#406080; ">{</span>self<span style="color:#308080; ">.</span>X<span style="color:#308080; ">:</span> X<span style="color:#406080; ">}</span><span style="color:#308080; ">)</span>
</pre>

#### 4.3.2. Instantiate and fit the Autoencoder:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Instantiate the Autoencoder</span>
model <span style="color:#308080; ">=</span> Autoencoder<span style="color:#308080; ">(</span><span style="color:#008c00; ">784</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">300</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># Fit the Autoencoder to the training data and plot the loss/cost function</span>
model<span style="color:#308080; ">.</span>fit<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
n_batches<span style="color:#308080; ">:</span> <span style="color:#008c00; ">656</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
epoch<span style="color:#308080; ">:</span> <span style="color:#008c00; ">0</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">562.033</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">200</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">88.770</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">400</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">70.945</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">600</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">63.730</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
epoch<span style="color:#308080; ">:</span> <span style="color:#008c00; ">1</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">63.642</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">200</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">62.351</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">400</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">60.547</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">600</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">58.097</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
epoch<span style="color:#308080; ">:</span> <span style="color:#008c00; ">28</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">53.676</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">200</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">51.435</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">400</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">52.713</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">600</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">51.097</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
epoch<span style="color:#308080; ">:</span> <span style="color:#008c00; ">29</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">49.435</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">200</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">51.667</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">400</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">50.340</span>
<span style="color:#400000; ">iter</span><span style="color:#308080; ">:</span> <span style="color:#008c00; ">600</span><span style="color:#308080; ">,</span> cost<span style="color:#308080; ">:</span> <span style="color:#008000; ">52.383</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Training completed successfully!
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

<img src="images/Loss function.png" width="1000"/>

#### 4.3.3 Use the trained Autoencoder to reconstruct the input images:¶

* Randomly select input images and reconstruct them using the trained Autoencoder:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># the number of reconstructed images</span>
num_reconstructed_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">10</span>
<span style="color:#200080; font-weight:bold; ">for</span> i <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>num_reconstructed_images<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">#----------------------------------------</span>
    <span style="color:#595979; "># Step 1: select a random input image:</span>
    <span style="color:#595979; ">#----------------------------------------</span>
    i <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>choice<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    x <span style="color:#308080; ">=</span> x_train<span style="color:#308080; ">[</span>i<span style="color:#308080; ">]</span>
    k <span style="color:#308080; ">=</span> y_train<span style="color:#308080; ">[</span>i<span style="color:#308080; ">]</span>
    im <span style="color:#308080; ">=</span> model<span style="color:#308080; ">.</span>predict<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span>x<span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#008c00; ">28</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#308080; ">)</span>
    <span style="color:#595979; ">#----------------------------------------</span>
    <span style="color:#595979; "># display the original and reconstructed </span>
    <span style="color:#595979; "># images:</span>
    <span style="color:#595979; ">#----------------------------------------</span>
    <span style="color:#595979; "># create a figure and set its axis</span>
    fig_size <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">7</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>
    <span style="color:#595979; "># create the figure </span>
    plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span>fig_size<span style="color:#308080; ">)</span>
    <span style="color:#595979; "># display the sample</span>
    plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">2</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>x<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#008c00; ">28</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> cmap<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'gray'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Original image with class: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>k<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">2</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">2</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>im<span style="color:#308080; ">,</span> cmap<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'gray'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Autoencoder reconstruction"</span><span style="color:#308080; ">,</span> fontsize <span style="color:#308080; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>


<table>
  <tr>
    <td> <img src="Reconstructed-samples-1-5.PNG"  width="1000"></td>
   </tr> 
   <tr>
    <td> <img src="Reconstructed-samples-6-10.PNG"  width="1000"></td>
  </td>
  </tr>
</table>

### 4.5. Part 5: Display a successful execution message:



<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># display a final message</span>
<span style="color:#595979; "># current time</span>
now <span style="color:#308080; ">=</span> datetime<span style="color:#308080; ">.</span>datetime<span style="color:#308080; ">.</span>now<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display a message</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>now<span style="color:#308080; ">.</span>strftime<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#1060b6; ">"</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

Program executed successfully on<span style="color:#308080; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">05</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">10</span> <span style="color:#008c00; ">03</span><span style="color:#308080; ">:</span><span style="color:#008c00; ">29</span><span style="color:#308080; ">:</span><span style="color:#008000; ">48.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>Goodbye!
</pre>


## 5. Analysis

* In view of the presented results, we make the following observations:
  * The Autoencoder reconstructions of the sample MNIST training images appear the same as the original images, to the uman eye. 
  * Autoencoder yield high-quality image loss reconstructions at high compression rate. 

## 6. Future Work

* We plan to explore the following related issues:
  * To estimate the compression ratio of the encoded images as compared to the original image
  * To explore implementing Variational Auto-Encoders (V-AE) and Generative Adversarial Networks (GANs) models to images from noise.
  * These more advanced models are expected to generated reconstructed images with even higher quality.

## 7. References

1. Kaggle. (Digit Recognizer: Learn computer vision fundamentals with the famous MNIST data. https://www.kaggle.com/c/digit-recognizer/data
2. Yann LeCun et. al. THE MNIST DATABASE of handwritten digits. http://yann.lecun.com/exdb/mnist/ 
3. JEREMY JORDAN. Introduction to autoencoders. https://www.jeremyjordan.me/autoencoders/ 
4. Keras. Building Autoencoders in Keras. https://blog.keras.io/building-autoencoders-in-keras.html Aditya Sharma. Autoencoder as a 5. Classifier using Fashion-MNIST Dataset. https://www.datacamp.com/community/tutorials/autoencoder-classifier-python 
6. Arvin Singh. Kushwaha Making an Autoencoder: Using Keras and training on MNIST. https://towardsdatascience.com/how-to-make-an-autoencoder-2f2d99cd5103 
7. Tensorflow. Introduction to Autoencoders. https://www.tensorflow.org/tutorials/generative/autoencoder 
8. Soumya Ghosh. Simple Autoencoder example using Tensorflow in Python on the Fashion MNIST dataset. https://medium.com/@connectwithghosh/simple-autoencoder-example-using-tensorflow-in-python-on-the-fashion-mnist-dataset-eee63b8ed9f1 
9. Adrian Rosebrock.Autoencoders with Keras, TensorFlow, and Deep Learning. https://www.pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/ 
10. mgid. Generate new MNIST digits using Autoencoderhttps://iq.opengenus.org/mnist-digit-generation-using-autoencoder/ 
11. Ali Abdelaal. Autoencoders for Image Reconstruction in Python and Keras. https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/ 
11. Jan Melchior.Autoencoder on MNIST. https://pydeep.readthedocs.io/en/latest/tutorials/AE_MNIST.html
