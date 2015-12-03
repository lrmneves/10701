# Run Theano using GPU on haopc
`THEANO_FLAGS='floatX=float32,device=gpu1,nvcc.fastmath=True' python testTheanoGPU.py`

1. If it says cannot find `nvcc`, add `nvcc` path to $PATH
`/usr/local/cuda-7.5/bin`

o2. If it says cannot find some libraries, run
`sudo ldconfig /usr/local/cuda-7.5/lib64`
