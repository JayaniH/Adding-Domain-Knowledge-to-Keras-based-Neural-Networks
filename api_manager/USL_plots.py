import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def f(n, s, k, l):
    return (1 + s*(n-1) + k*n*(n-1))

def f1(n, s, k, l, L1):
    print(L1)
    return (f(n, s, k, l)/l) + L1

def f2(n, s, k, l, L1):
    return f(n, s, k, l) * (1 + L1) / l

def f3(n, s, k, l, m):
    return f(n, s, k, l) / (l + m)


n = np.arange(0, 100, 1)
s = 0.1
k = 0.1
l = 1

for L1 in [-100, 0, 100]:
    y = f1(n, s, k, l, L1)
    plt.plot(n, y, label='L1 = '+str(L1))

plt.xlabel('concurrent_users')
plt.ylabel('latency')
plt.ylim(bottom=0)
plt.legend()
plt.show()
# plt.savefig('../../Plots/_api_manager/18_domain_model_minimization_eq2_regularization_param_10000a_100b_10a1/' + str(i+1) + '_msg_size.png')
plt.close()

for L1 in [-2, -1, -0.5, 0, 2]:
    y = f2(n, s, k, l, L1)
    # print(y2)
    plt.plot(n, y, label='L1 = '+str(L1))

plt.xlabel('concurrent_users')
plt.ylabel('latency')
plt.legend()
plt.show()
# plt.savefig('../../Plots/_api_manager/18_domain_model_minimization_eq2_regularization_param_10000a_100b_10a1/' + str(i+1) + '_msg_size.png')
plt.close()

for m in [-2, -1, -0.5, 0, 2]:
    y = f3(n, s, k, l, m)
    # print(y1)
    plt.plot(n, y, label='m = '+str(m))

plt.xlabel('concurrent_users')
plt.ylabel('latency')
plt.legend()
plt.show()
# plt.savefig('../../Plots/_api_manager/18_domain_model_minimization_eq2_regularization_param_10000a_100b_10a1/' + str(i+1) + '_msg_size.png')
plt.close()
