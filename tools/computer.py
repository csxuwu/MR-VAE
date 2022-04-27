

import numpy as np

gamma = [0.5,0.8,1.0,1.5,2.0,2.5,3.0]

for i in range(len(gamma)):
    sigma_e = np.power(np.e,gamma[i])
    sigma_linear = 10 * gamma[i]
    print('gamma:{}     sigma_e:{}     sigma_linear:{} '.format(gamma[i],sigma_e,sigma_linear))