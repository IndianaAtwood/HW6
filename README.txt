Current state of HW5b as of Sun, Nov. 9 at 11:00 am:

- I used our code from HW5a and Quince and I's HW2 (is pretty efficient) into hw5b.py

- Runs the ReAntics game like in HW2, but when a move is chosen (by best utility),
it calls a function that runs forward propogation, compares the output to the utility
function, and then does backward propogation based on the error (one forward/back per move).
Then, it saves the weights into ANN_weights.txt (not readable, unfortunately).

- I am hoping that this is training effectively, but I can't be absolutely sure. My plan
is to train it a bunch, and then get rid of the utility function and just use the ANN (right?)

- Text me if you have any questions