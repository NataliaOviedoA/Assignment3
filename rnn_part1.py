
import load_dataset as data

#If final is true, the function returns the canonical test/train split with 25 000 reviews in each.
#If final is false, a validation split is returned with 20 000 training instances and 5 000
#validation instances.

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = data.load_imdb(final=False)

# The return values are as follows:
# x_train A python list of lists of integers. Each integer represents a word. Sorted from short to long.
# y_train The corresponding class labels: 0 for positive, 1 for negative.
# x_val Test/validation data. Laid out the same as x_train.
# y_val Test/validation labels
# i2w A list of strings mapping the integers in the sequences to their original words. i2w[141] returns the string containing word 141.
# w2i A dictionary mapping the words to their indices. w2i['film'] returns the index for the word "film".

# To have a look at your data (always a good idea), you can convert a sequence from indices to words as follows
print([i2w[w] for w in x_train[141]])

# To train, you'll need to loop over x_train and y_train and slice out batches. 
# Each batch will need to be padded to a fixed length and then converted to a torch tensor. 
# Implement this padding and conversion to a tensor
