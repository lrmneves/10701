#Caffe Setup

**This is only one of the many ways to install Caffe and pycaffe on your computer, you might wanna take a look at the [Caffe official installation instruction](http://caffe.berkeleyvision.org/installation.html) and make your custom installation**

1. install the pre-requisites

    ```
    brew install homebrew/science/openblas homebrew/science/opencv
    brew install protobuf gflags glog hdf5 leveldb lmdb
    brew install boost --with-python
    brew install boost-python
    ```

2. clone Caffe from github

    ```
    git clone https://github.com/BVLC/caffe.git
    ```

3. modify Makefile.config.example:
    1. uncomment `CPU_ONLY := 1`
    2. set `BLAS := open`
    3. *Homebrew puts openblas in a directory that is not on the standard search path*, uncomment the `BLAS_INCLUDE` and `BLAS_LIB` after that line
    4. modify the `PYTHON_INCLUDE` a couple lines below according to your local settings
    5. *Homebrew installs numpy in a non standard path*, uncomment `PYTHON_INCLUDE` after that line
    6. modify `PYTHON_LIB` below to

        ```
        PYTHON_LIB += $(dir $(dir $(shell python -c 'import numpy.core; print(numpy.core.__file__)')))/lib
        ```
        
4. Create `Makefile.config`

    ```
    cp Makefile.config.example Makefile.config
    ```
    
5. Export pycaffe to your `PYTHONPATH`

    ```
    export PYTHONPATH=/path/to/caffe/python:$PYTHONPATH
    ```
    
6. Brew!

    ```
    make all -j 4
    make test
    make runtest
    s
    ```

7. run

    ```
    python -c "import caffe; print caffe.TEST"
    ```

    If you see a `1`, then Caffe and pycaffe are installed in your system, have fun!
    
    If you you run the code and see some errors about missing libraries, you might wanna try
    
    ```
    cd /path/to/your/caffe/python
    for req in `cat requirements.txt`; do sudo pip install $req; done
    ```

#Reference
[1] [Caffe official installation instruction](http://caffe.berkeleyvision.org/installation.html)

[2] [Caffe official installation instruction for OS X](http://caffe.berkeleyvision.org/install_osx.html)

[3] [How to install Caffe on Mac OS X 10.10 for dummies (like me)](http://hoondy.com/2015/04/03/how-to-install-caffe-on-mac-os-x-10-10-for-dummies-like-me/)

#Other CNN libraries (with pre-trained CNN)
1. [VGG Convolutional Neural Networks Practical - Matlab](http://www.robots.ox.ac.uk/~vgg/practicals/cnn/)
2. [MatConvNet - Matlab](http://www.vlfeat.org/matconvnet/pretrained/#imagenet-ilsvrc-classification)
3. [VGG deepeval-encoder - Matlab (OS X not supported)](http://www.robots.ox.ac.uk/~vgg/software/deep_eval/)
4. [CCV](http://libccv.org/doc/doc-convnet/)
