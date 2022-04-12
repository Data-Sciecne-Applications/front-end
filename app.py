from flask import Flask, render_template, request, flash, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np
# import enchant
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.cluster import KMeans 
import PyPDF2
import sys
import time
import spacy
import warnings
import os
warnings.filterwarnings('ignore')
sys.path.append("..")
from functions import helper_functions as hf, similarity_functions as sf

app = Flask(__name__)
UPLOAD_FOLDER = os.path.dirname(app.instance_path) + '/resumes'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
# db = SQLAlchemy(app)

# class Resume(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     file_path = db.Column(db.String(200), nullable=False)

#     def __repr__(self):
#         return '<Task %r>' % self.id

def process_resume(file_path, file_name):
    # Load the en_core_web_sm model
    nlp = spacy.load('en_core_web_sm')
    stopwords = spacy.lang.en.stop_words.STOP_WORDS

    # Load data
    df_jobs = pd.read_csv("./data/df_job_final.csv", usecols=['title', 'department', 'description_combined'])
    df_resume = pd.read_csv("./data/data_resume_cc.csv", usecols=['Category', 'Resume_c'])
    # department mapping between job and resume labels
    THRESHOLD = 50
    df_jobs = hf.get_map_category(df_jobs, 'department', THRESHOLD )
    df_resume = hf.get_map_category(df_resume, 'Category', THRESHOLD)
    df_jobs_nan = df_jobs[df_jobs['department'].isna() == True]
    df_jobs = df_jobs[df_jobs['department'].isna() == False]
    # load models
    vec = hf.load_tfidf('./ml_models/tfidf_job.pkl')
    svm_clf = hf.load_tfidf('./ml_models/tfidf_clf.pkl')
    kmeans = hf.load_tfidf('./ml_models/tfidf_cluster.pkl')
    df_jobs['cluster'] = kmeans.labels_

    # start processing resume
    start = time.time()
    pdffileobj = open(f'{file_path}/{file_name}','rb')
    pdfreader = PyPDF2.PdfFileReader(pdffileobj)
    pages = pdfreader.numPages
    resume = ''
    for i in range(0, pages):
        pageobj = pdfreader.getPage(i)
        resume = resume + pageobj.extractText() + ' '

    ##lemmatize Resume
    lemmas = [token.lemma_ for token in nlp(resume)]
    resume = [lemma.lower() for lemma in lemmas if lemma.isalpha() and lemma not in stopwords or lemma == '-PRON-']
    resume = ' '.join(resume)

    pred_department = svm_clf.predict([resume])[0]
    resume_matrix = vec.transform([resume])[:, :15240]
    pred_cluster = kmeans.predict(resume_matrix)[0]

    jobs_top_n = hf.get_top_n_jobs_from_clf(df_jobs=df_jobs,
                            pred_department=pred_department,
                            resume=resume,
                            vec=vec,
                            sim_func=sf.cal_cosine_similarity)
    rec_clf = df_jobs.iloc[jobs_top_n]
    rec_clf_filtered = rec_clf[rec_clf['department'] == pred_department]

    jobs_top_n = hf.get_top_n_jobs_from_cluster(df_jobs, pred_cluster, resume, vec, sf.cal_cosine_similarity)
    rec_cluster = df_jobs.iloc[jobs_top_n]
    rec_cluster_filtered = rec_cluster[rec_cluster['department'] == pred_department]

    # assume last 50% of the resumes got offers
    applicant_pool_with_offer = df_resume[:(int)(len(df_resume)*0.5)]
    application_pool = applicant_pool_with_offer[applicant_pool_with_offer['Category'] == pred_department]
    # build resume similarity matrix
    temp = application_pool['Resume_c'].append(pd.Series(resume))
    matrix =vec.transform(temp)
    term_matrix = matrix.todense()
    cossim = sf.cal_cosine_similarity(term_matrix)
    # get similar resumes based on given resume
    index_similar_applicant = np.asarray(cossim[-1][np.where(cossim[-1] < 1)]).argsort()[::-1][:10]
    cf_jobs = hf.get_top_n_jobs_from_cf(df_jobs, df_resume, index_similar_applicant, svm_clf, vec, sf.cal_cosine_similarity, 5)
    # recommend jobs based on similar resumes
    rec_from_cf = df_jobs.iloc[cf_jobs]
    rec_from_cf_filtered = rec_from_cf[rec_from_cf['department'] == pred_department]

    list_filtered_rec = [rec_clf_filtered, rec_cluster_filtered, rec_from_cf_filtered]
    final_rec = pd.concat(list_filtered_rec)
    final_rec = final_rec.drop_duplicates()
    end = time.time()

    return final_rec, end-start
##################################################
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        file = request.files['upload_resume']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            recommendations, time_cost = process_resume(app.config['UPLOAD_FOLDER'], filename)
            
        return render_template('index.html',
        title=recommendations['title'],
        description=recommendations['description_combined'],
        time_cost=time_cost,
        zip=zip)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=5001)