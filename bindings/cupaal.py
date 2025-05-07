import pybind11
# import cupaal_bindings


# matrix = numpy.array
# labelling = ['a', 'b', 'c']
# inittial_state = matrix(first column)


# observations
sequences = [['a', 'b', 'c'],
             ['d', 'e', 'f']]
times = [1,3]

def make_cupaal_observations(sequences, times):
    """
    make a list that copies the sequences times the number of sequences
    """
    observations = []
    for i in range(len(sequences)):
        number = times[i]
        for j in range(number):
            observations.append(sequences[i])
    return observations


print(str(make_cupaal_observations(sequences, times)))







#cuppal_bindings.bw_wrapping_function()