#import jajapy as ja
import numpy as np
import ctypes
import os


lib = ctypes.CDLL('./test.so')

# lav function i Cupaal python script, som er wrapper omkring de ting i Jajapy og giver  
# LoadPrism, Instatiate, fit_Parameters, vil vi ikke kalde i CuPaal. Istedet vil vi kalde en enkelt som gør det hele. Til det skal den bruge en path, random initial varialbes/liste over parametre som skal fittes, vores observations.

def fitParameters(path, training_set, parameters_to_estimate):

    training_set = training_set.astype(np.float64)
    n_obs, n_states = training_set.shape
    lib.fitParameters(path, training_set, parameters_to_estimate)
    #Call c++ function 
    #BW(path, training_set, parameters_to_estimate)
    


if __name__ == '__main__':
    path = "examples/materials/philosophers1.sm"
    training_set = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    parameters_to_estimate = ["alpha", "beta", "zeta"]
    fitParameters(path, training_set, parameters_to_estimate)
    print("Hello World")