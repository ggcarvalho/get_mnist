# Read Me

To save the MNIST dataset in .npy format run

        python get_mnist.py <threshold>

If `threshold < 0` the dataset will be saved as is, otherwise the images will be saved as binary images with the given threshold (eg, if `threshold` = 0.75, any value less than or equal to 0.75 will be set to zero and any value grater than 0.75 will be set to 1).

