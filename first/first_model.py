import numpy as np    #numpy 1.15.2
import pandas as pd   #pandas 0.23.4
import networkx as nx #networkx 2.2
from datetime import datetime

#defining the predict function
def predict(predict_ticker, predict_date, orig_graph, num_ripples, timeseries_data, moved_ticker = None):
    #initialise variables
    p_orig   = timeseries_data[predict_ticker][str(predict_date)] #original price
    p_target = timeseries_data[predict_ticker].shift(-1)[str(predict_date)] #price we are trying to predict
    R        = np.asarray(nx.adjacency_matrix(orig_graph).todense()) + np.diag(np.array([1]*len(orig_graph.nodes))) #influence matrix
    D        = np.zeros(R.shape) #change matrix

    #setting prices in graph
    for node in orig_graph.nodes:
        orig_graph.node[node]["px"] = timeseries_data[node][str(predict_date)]

    #define original price variables
    G_t_minus_2 = orig_graph.copy()
    p_t_minus_2 = np.array([G_t_minus_2.node[x]["px"] for x in G_t_minus_2.nodes])
    G_t_minus_1 = orig_graph.copy()

    #what has moved?
    if moved_ticker == None:
        moved_ticker = timeseries_data.news[predict_date].keys()
    else:
        moved_ticker = [moved_ticker]

    for ticker in moved_ticker:
        moved_p_orig = timeseries_data[ticker][str(predict_date)] #original price
        moved_p_next = timeseries_data[ticker].shift(-1)[str(predict_date)] #next day price
        moved_change = (moved_p_next - moved_p_orig) / moved_p_orig
        G_t_minus_1.node[ticker]["px"] *= moved_change + 1

    p_t_minus_1 = np.array([G_t_minus_1.node[x]["px"] for x in G_t_minus_1.nodes])

    #apply the ripple equations
    for iteration in range(num_ripples):

        #create change matrix D for this iteration
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if i == j:
                    D[i][j] = 1
                elif R[i][j] == 1:
                    D[i][j] = (p_t_minus_1[i] - p_t_minus_2[i]) / p_t_minus_2[i]
                else:
                    0

        #run one iteration of ripple equations
        p_t = np.diag(np.matmul(R,D)) * p_t_minus_1

        #define new price variables
        p_t_minus_2 = p_t_minus_1.copy()
        p_t_minus_1 = p_t.copy()

    predict_ticker_index = list(G.nodes).index(predict_ticker)
    p_predict = p_t[predict_ticker_index]

    return p_predict, p_orig, p_target



if(__name__ == "__main__"):
    start = datetime.now()

    #load data
    tf = pd.read_csv("tf.csv", index_col = 0)
    tf.news = tf.news.apply(eval)
    G  = nx.read_gpickle("G.pickle")

    #first example
    print("****** FIRST EXAMPLE ******")
    moved_ticker    = "AAPL"
    predict_ticker  = "TWTR"
    predict_date    = "2018-07-11"
    orig_graph      = G
    num_ripples     = 5
    timeseries_data = tf
    print(predict(predict_ticker, predict_date, orig_graph, num_ripples, timeseries_data, moved_ticker = None))
    print("(prediction, original, actual)")
    print("")

    #second example
    print("****** SECOND EXAMPLE ******")
    predict_ticker  = "TWTR"
    orig_graph      = G
    num_ripples     = 5
    timeseries_data = tf
    acc_count       = 0
    for date in timeseries_data.index:
        predict_date   = str(date)[:10]
        prediction, original, actual = predict(predict_ticker, predict_date, orig_graph, num_ripples, timeseries_data, moved_ticker = None)
        if (prediction > original) == (actual > original) : acc_count += 1

    print("Accuracy", acc_count/ len(timeseries_data.index))
    print("")

    print("Time:", datetime.now() - start)
