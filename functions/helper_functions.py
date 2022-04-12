import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans 
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import time
from sklearn.pipeline import Pipeline


def save_tfidf(path, tfidf_vec):
    with open(path, 'wb') as fw:
        pickle.dump(tfidf_vec, fw)
        fw.flush()
    
def load_tfidf(path):
    f_vec = open(path, 'rb')
    vec = pickle.load(f_vec, encoding='utf-8')
    return vec

def get_map_category(df, col_name, threshold):
    df_ = df.copy(deep=True)
    map_ = build_category_map() 
    # map items if in map_
    for i, item in enumerate(df_[col_name]):
        if item in map_:
            df_[col_name][i] = map_[item]
    # get counts for unique item
    counts = df_[col_name].value_counts()
    # get used items from the map
    useful = counts > threshold
    useful = [key for key in useful.keys() if useful[key] == True]
    df_ = df_[df_[col_name].isin(useful) == True]
    return df_


def build_category_map():
    category_map = dict()

    SALE = 'sales'
    ENGINEERING = 'engineering'
    MARKETING = 'marketing'
    OPEARTION = 'operation'
    IT = 'it'
    HEALTH = 'health'
    RESEARCH = 'research'
    CONSTRUCTION = 'construction'
    EDUCATION = 'education'
    CUSTOMER = 'customer'

    category_map = dict.fromkeys(['sales', 'retail'], SALE)
    category_map.update(dict.fromkeys(['engineering'], ENGINEERING))
    category_map.update(dict.fromkeys(['operations'], OPEARTION))
    category_map.update(dict.fromkeys(['it', 'development', 'product', 'information technology', 'design', 'technology', 'tech', 'designing'], IT))
    category_map.update(dict.fromkeys(['customer service'], CUSTOMER))
    category_map.update(dict.fromkeys(['r&d', 'business services', 'aerospace & defense', 'government'], RESEARCH))
    category_map.update(dict.fromkeys(['health care', 'biotech & pharmaceuticals', 'health & fitness'], HEALTH))
    category_map.update(dict.fromkeys(['construction, repair & maintenance'], CONSTRUCTION))
    category_map.update(dict.fromkeys(['education'], EDUCATION))
    return category_map


def get_top_n_jobs_from_clf(df_jobs, pred_department, resume, vec, sim_func, n=10):
    potential_jobs = df_jobs[df_jobs['department'] == pred_department]
    temp = potential_jobs['description_combined'].append(pd.Series(resume))
    matrix = vec.transform(temp)
    term_matrix = matrix.todense()
    sim_matrix = sim_func(term_matrix)
    return np.asarray(sim_matrix[-1][np.where(sim_matrix[-1] < 1)]).argsort()[::-1][:n]

def get_top_n_jobs_from_cluster(df_jobs, pred_cluster, resume, vec, sim_func, n=10):
    potential_jobs = df_jobs[df_jobs['cluster'] == pred_cluster]
    temp = potential_jobs['description_combined'].append(pd.Series(resume))
    matrix = vec.transform(temp)
    term_matrix = matrix.todense()
    sim_matrix = sim_func(term_matrix)
    return np.asarray(sim_matrix[-1][np.where(sim_matrix[-1] < 1)]).argsort()[::-1][:n]

def get_top_n_jobs_from_cf(df_jobs, df_resume, index_similar_applicant, clf, vec, sim_func, n=1):
    cf_jobs = []
    for index in index_similar_applicant:
        similar_applicant = df_resume.iloc[index]
        pred_similar_applicant = clf.predict([similar_applicant['Resume_c']])[0]

        potential_jobs = df_jobs[df_jobs['department'] == pred_similar_applicant]
        temp = potential_jobs['description_combined'].append(pd.Series(similar_applicant['Resume_c']))
        matrix = vec.transform(temp)
        term_matrix = matrix.todense()
        sim_matrix = sim_func(term_matrix)
        job = np.asarray(sim_matrix[-1][np.where(sim_matrix[-1] < 1)]).argsort()[::-1][:n]
        cf_jobs.extend(job)
    return cf_jobs


def get_classification_model_performance(estimator, transformer, x_train, x_test, x_val, y_train, y_test, y_val, n=10)-> dict():
    start = time.time()
    clf = Pipeline([
        ('tf', transformer),
        ('clf', estimator)
    ])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    done = time.time()
    accuracy = np.mean(pred == y_test)
    cross_val = cross_val_score(estimator=clf, X=x_val, y=y_val, cv=n)
    print(f"accuracy: {accuracy}, 10-fold: {np.mean(cross_val)}")

    cf_report = classification_report(y_test, pred, zero_division=0, output_dict=True)

    performance = dict()
    performance['estimator'] = estimator
    performance['accuracy'] = accuracy
    performance['cv10'] = np.mean(cross_val)
    performance['precision'] = cf_report['macro avg']['precision']
    performance['recall'] = cf_report['macro avg']['recall']
    performance['f1_score'] = cf_report['macro avg']['f1-score']
    performance['time_cost'] = done - start

    return performance