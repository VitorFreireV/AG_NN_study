import numpy as np
def fo(new_state2):
    row = int(new_state2/4)
    col = int(new_state2%4)
    result = np.sqrt((4-row)**2 + (4-col)**2)
    result = 1/result
    #print("estado: ", new_state2, '\nFo:', result)
    return result

for i in range(0, 16):
    print(i,fo(i))