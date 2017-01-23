# In this exercise we'll examine a learner which has high variance, and tries to learn
# nonexistant patterns in the data.
# Use the learning curve function from sklearn.learning_curve to plot learning curves
# of both training and testing error.

from sklearn.tree import DecisionTreeRegressor

# import matplotlib as mpl
# Tell Matplotlib to use the correct backend
# mpl.use('MacOSX')

import matplotlib.pyplot as plt

# to run matplotlib and show the plots on mac, apparently I need to use pylab
# from pylab import *

from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import KFold
from sklearn.metrics import explained_variance_score, make_scorer
import numpy as np

# Global variables
# Set the learning curve parameters; you'll need this for learning_curves
size = 1000
cv = KFold(size,shuffle=True)
score = make_scorer(explained_variance_score)

# main routine
def main():

    # Create a series of data that forces a learner to have high variance
    X = np.round(np.reshape(np.random.normal(scale=5,size=2*size),(-1,2)),2)
    y = np.array([[np.sin(x[0]+np.sin(x[1]))] for x in X])
    plot_curve(X,y)


# function to plot the learning curves
def plot_curve(X,y):
    reg = DecisionTreeRegressor()
    reg.fit(X,y)
    print "Regressor score: {:.4f}".format(reg.score(X,y))
    
    # TODO: Use learning_curve imported above to create learning curves for both the
    #       training data and testing data. You'll need 'size', 'cv' and 'score' from above.
    
    train_sizes, train_scores, test_scores = learning_curve(
        DecisionTreeRegressor(random_state=0), X, y, cv=cv, scoring = score)
    train_scores_mean=np.mean(train_scores, axis=1)
    test_scores_mean=np.mean(test_scores, axis=1)
    
    # TODO: Plot the training curves and the testing curves
    #       Use plt.plot twice -- one for each score. Be sure to give them labels!
    
    # plot the train scores learning curve
    plt.plot(train_sizes ,train_scores_mean,'-o',color='r',label="train_scores_mean")

    #  plot the test scores learning curve
    plt.plot(train_sizes,test_scores_mean ,'-o',color='g',label="test_scores_mean")
    
    # Plot aesthetics
    plt.ylim(-0.1, 1.1)
    plt.ylabel("Curve Score")
    plt.xlabel("Training Points")
    plt.legend(bbox_to_anchor=(1.1, 1.1))
    # save the plot
    plt.savefig('scores_mean')
    # plt.show()

    # Block the run so we can see the plot in the GUI
    plt.show(block=True)
    # pylab.show()

if __name__ == '__main__':
    main()