from datetime import datetime
start = datetime.now()
print("****** LOADING MODULES AND DATA ******", "\n")
print("Starting at:", start)


#nlp and graph modules
import networkx as nx
import pandas as pd
import numpy as np
import json, os, re
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer, PatternAnalyzer
from collections import Counter

filename = "results" + str(datetime.now()) + ".txt"


print("Time to load modules:", datetime.now() - start)

################# ################# ################# #################
""" Support Functions """
start = datetime.now()
print("Starting at:", start)

#symbols, names and tickers lists
with open('symbols.json') as json_file:
    symbols = json.load(json_file)
tickers = {}
for sym in symbols:
    tickers[sym["symbol"]] = sym
names = {v["name"]: k for k, v in tickers.items()}
print("symbols is a list of tickers")
print("tickers is a dict of tickers and names")
print("names is a dict of names and tickers")

#table of correlations
tf = pd.read_pickle("stocks2.pickle")
rf = pd.read_pickle("returns.pickle")
print("tf is a table of prices")
print("rf is a table of typical returns")

#names lookup table
lf = pd.read_csv("lookup.csv", index_col = 0, header = None, names = ["Ticker" , "one" , "two" , "three" , "four"])
lookup = {}
for col in lf.columns:
    if col == "Ticker":
        continue
    for i in range(len(lf[col])):
        name = lf[col][i]
        if type(name) != np.float:
            lookup[name] = lf["Ticker"][i]
print("lf is a lookup table for stock names")

#load model data
cols = ["headline", "body", "nlp", "entities", "dateCreated"]
df = pd.read_csv("model.csv")
for column in ["creator", "keywords", "body", "entities"]:
    df[column] = df[column].apply(eval)
print("df is model data")

print("Time to load data:", datetime.now() - start)

#defining all the functions needed to process the data

start = datetime.now()
print("Starting at:", start)
#create a dictionary of synonyms
syn_dict = {"united_states" : "us",
           "federal_reserve" : "fed",
           "us_federal_reserve" : "fed",
           "european_central_bank" : "ecb",
           "goldman" : "goldman_sachs"}

#define a function to preprocess entities
def clean_entities(ents, with_type = True):
    new_ents = []
    for ent in ents:
        new_ent = []
        new_ent = [ent[0].lower(), ent[1]]
        new_ent[0] = re.compile("the ").sub("", new_ent[0])
        new_ent[0] = re.compile("[^0-9a-zA-Z #]").sub("", new_ent[0])
        new_ent[0] = re.compile(" ").sub("_", new_ent[0])

        if new_ent[0] in syn_dict.keys():
            new_ent[0] = syn_dict[new_ent[0]]

        #return list of tuples if you want to include the type, otherwise just return list of entities
        if with_type == False:
            new_ents.append(new_ent[0])
        else:
            new_ents.append(tuple(new_ent))
    return new_ents

def preprocess_entities():
    verbose = False
    ent_list = []
    error_count = 0
    ent_series = df.entities[:]
    for row in ent_series:
        if "E" in row:
            error_count += 1
        elif row != []:

            row = clean_entities(row)
            ent_list.extend(row)
    if verbose:
        print("number of articles:", ent_series.shape[0])
        print("number of articles with errors:", error_count)
        print("number of entities:", len(ent_list))
    top = Counter(ent_list).most_common(300)

def search_entities(ent_list):
    return [x[0] for x in clean_entities(ent_list) if x[0] in lookup.keys()]

def calc_sentiment(headline):
    sentiment = TextBlob(headline, analyzer=PatternAnalyzer()).sentiment[0]
    if sentiment > 0.1:
        return 1
    elif sentiment < -0.1:
        return -1
    else:
        return 0

def calc_sentiment_precise(headline):
    sentiment = TextBlob(headline, analyzer=PatternAnalyzer()).sentiment[0]
    if   sentiment >  0.5: return  0.10
    elif sentiment >  0.1: return  0.05
    elif sentiment < -0.5: return -0.10
    elif sentiment < -0.1: return -0.05
    else: return 0

def calc_sentiment_returns(headline_list):
    if type(headline_list) == str: return {}
    headline = headline_list[0]
    tickers = headline_list[1:]
    sentiment_dict = {}
    sentiment = TextBlob(headline, analyzer=PatternAnalyzer()).sentiment[0]
    sentiment = round(sentiment * 4) / 4
    for ticker in tickers:
        try:
            sentiment_dict[ticker] = rf[ticker].loc[sentiment] / 100
        except:
            sentiment_dict[ticker] = sentiment / 100
    return sentiment_dict

def lookup_name(names):
    return list(set([lookup[name] for name in names]))

def convert_date(dateandtime):
    d = dateandtime.date()
    if dateandtime.weekday() == 6:
        d = d + pd.DateOffset(days = 1)
    elif dateandtime.weekday() == 5:
        d = d + pd.DateOffset(days = 2)
    elif d == pd.to_datetime("4th July 2018").date():
        d = d + pd.DateOffset(days = 1)
    return d

def pull_date(dateandtime):
    return str(dateandtime.date()) + " " + str(dateandtime.day_name())

#build the adjacency matrix and graph
#returns 1 if correlation over threshold
def simple_a(x, thresh):
    return (abs(x) >= thresh) * 1 & (abs(x - 1.0) > 0.00001) * 1

#returns correlation if over threshold
def weighted_a(x, thresh):
    if abs(x) <= thresh or abs(x) == 1: return 0.0
    else: return x

def build_graph(date, thresh = 0.8, timeseries_data = tf, adjacency_func = weighted_a):
    idx              = timeseries_data.index.get_loc(pd.to_datetime(date))
    correlations     = timeseries_data.iloc[idx - 30:idx].corr()
    adjacency_matrix = correlations.applymap(lambda value : adjacency_func(value, thresh))
    G                = nx.from_numpy_matrix(adjacency_matrix.values)
    G                = nx.relabel_nodes(G , dict(zip(G.nodes() , adjacency_matrix.index)))
    return G

print("Time to define functions needed to prep data:", datetime.now() - start)


start = datetime.now()
print("Starting at:", start)

#configure table holding the final data that the model needs
def config_timeseries(news_data, price_data, start_date, end_date, sentiment_func):
    verbose = False
    mf = news_data.copy()
    if verbose: print("mf.shape before", mf.shape)
    preprocess_entities()
    mf["date"]      = pd.to_datetime(mf.dateCreated).apply(convert_date)
    mf              = mf[(mf.date >= start_date) & (mf.date < end_date) ]
    mf["my_ents"]   = mf.entities.apply(search_entities).apply(lookup_name)
    mf["sentiment"] = mf.headline.apply(lambda x : [x]) + mf.my_ents
    mf["sentiment"] = mf.sentiment.apply(sentiment_func)

    if verbose: print("mf.shape after", mf.shape)

    #combining news and price data into tf
    date_news = {} #blank dict
    for d in mf.date: #one key for each date
        date_news[d] = []
    for i in range(mf.shape[0]): #combine all sentiment of each news article on that day
        date_news[mf.date.iloc[i]].append(mf.sentiment.iloc[i])
    for date in date_news.keys(): #combine all dicts per day into one
        result = {}
        for dictionary in date_news[date]:
            result.update(dictionary) #need a better way to merge dictionaries, averaging maybe?
        date_news[date] = result

    price_data         = price_data.loc[start_date:end_date].copy()
    price_data["news"] = pd.Series(date_news)

    return price_data

print("Time to prepare model data:", datetime.now() - start)
middle = datetime.now()

################# ################# ################# #################
""" Predict Function """
def predict(predict_date, num_ripples, thresh, adjacency_func, timeseries_data, moved_ticker = None,
            use_sentiment = False, dampening = 0):

    #initialise variables
    orig_graph      = build_graph(date = predict_date, thresh = thresh, timeseries_data = timeseries_data,
                                  adjacency_func = adjacency_func)
    timeseries_data = timeseries_data[list(orig_graph.nodes) + ["news"]]
    p_orig          = timeseries_data.loc[str(predict_date)] #original prices
    p_target        = timeseries_data.shift(-1).loc[str(predict_date)] #prices we are trying to predict
    R               = np.asarray(nx.adjacency_matrix(orig_graph).todense())                       + np.diag(np.array([1]*len(orig_graph.nodes))) #influence matrix
    D               = np.zeros(R.shape) #change matrix

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

    #how much has it moved?
    moved_ticker = [ticker for ticker in moved_ticker if ticker in orig_graph.nodes()]
    for ticker in moved_ticker:
        moved_p_orig = timeseries_data[ticker][str(predict_date)] #original price
        if use_sentiment:
            moved_change = timeseries_data.news[predict_date][ticker]
            moved_p_next = moved_p_orig * (moved_change + 1)
        else:
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
                elif R[i][j] != 0: #careful, might need to change when you change the adjacency matrix!
                    #dealing with divide by 0 errors
                    if abs(p_t_minus_2[i]) < 0.00001:
                        D[i][j] = 0.0
                    else:
                        dampening = 1 / (1 + iteration * dampening)
                        D[i][j]   = dampening * (p_t_minus_1[i] - p_t_minus_2[i]) / p_t_minus_2[i]
                else:
                    0

        #run one iteration of ripple equations
        p_t = np.diag(np.matmul(R,D)) * p_t_minus_1

        #define new price variables
        p_t_minus_2 = p_t_minus_1.copy()
        p_t_minus_1 = p_t.copy()

    zipped = zip(p_t, p_orig, p_target)
    p_predict = dict(zip(orig_graph.nodes, zipped))

    return p_predict

print("Time to prep predict function:", datetime.now() - middle)

################# ################# ################# #################
""" EVALUATION """
if(__name__ == "__main__"):

    end_date        = pd.to_datetime("1st September 2018")
    start_date      = end_date - pd.DateOffset(months = 6)
    start = datetime.now()
    print("Starting at:", start)
    timeseries_data = config_timeseries(news_data = df[cols], price_data = tf, start_date = start_date, end_date = end_date, sentiment_func = calc_sentiment_returns)
    adjacency_func  = weighted_a


    print("Time to build variables:", datetime.now() - start)

    print("\n", "****** BEGINNING TEST ******")
    middle = datetime.now()

    for dampening in [x/10.0 for x in range(0, 30, 2)]:
        for thresh in [x/10.0 for x in range(0, 10, 1)]:
            for num_ripples in range(1, 10, 1):
                total_acc   = 0
                total_moved = 0

                for date in timeseries_data.index[:]:
                    acc_count   = 0
                    moved_count = 0
                    predict_date = str(date)[:10]
                    predictions  = predict(predict_date = predict_date, num_ripples = num_ripples, thresh = thresh,
                                           adjacency_func = adjacency_func, timeseries_data = timeseries_data,
                                           dampening = dampening, moved_ticker = None, use_sentiment = True)
                    #print(predictions)
                    for prediction in predictions.values():
                        predicted   = prediction[0]
                        original    = prediction[1]
                        actual      = prediction[2]
                        pred_change = (predicted - original) / original
                        act_change  = (actual    - original) / original
                        if abs(pred_change) > 0.0001:
                            moved_count += 1
                            if (pred_change > 0) == (act_change > 0):
                                acc_count += 1
                    total_acc   += acc_count
                    total_moved += moved_count
                    if moved_count == 0: acc = "n/a"
                    else:
                        acc = "{:.2f}".format(acc_count / moved_count)
                        #print("accuracy", acc_count/moved_count)



                print("num_ripples:", num_ripples, "counts:", total_acc, total_moved,
                      "accuracy:", "{:.2f}".format(total_acc / total_moved), "\n")

                result = (dampening, thresh, num_ripples, total_acc, total_moved, total_acc / total_moved)

                with open(filename, "a") as f:
                    f.write(str(result) + "\n")

                with open(filename, "r") as f:
                    if len(f.readlines()) > 1:
                        print("****** EXITING******")
                        print("****** PLEASE CHANGE .py FILE IF YOU REQUIRE LONGER TEST ******")
                        raise SystemExit

            print("Time to predict and evaluate:", datetime.now() - middle)
