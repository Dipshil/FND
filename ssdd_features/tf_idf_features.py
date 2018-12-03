from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pattern.en import tag as pos_tagger
import pandas as pd
import numpy as np
import csv
import argparse
import os
import pickle
import nltk
from ast import literal_eval

parser=argparse.ArgumentParser()
parser.add_argument('--mode',type=str,choices=['train','test','valid'],help="mode")
parser.add_argument('--features',type=str,choices=['word','shallow','deep'],help="features to use")
parser.add_argument('--data_dir',type=str,help="directory containing data files")
args=parser.parse_args()


if __name__ == '__main__':
	filename = os.path.join(args.data_dir,str(args.mode)+'.tsv')
	df=pd.read_csv(filename,header=None,sep='\t')
	df=df.iloc[:,0:3]
	print(df.iloc[:,2].head())
	if args.features == 'word':
		tfidf_vectorizer=TfidfVectorizer(ngram_range=(1,1))
		tfidf_matrix = tfidf_vectorizer.fit_transform(df.iloc[:,2])
		df_unigram=pd.DataFrame(list(tfidf_matrix))
		df_unigram.to_csv('word_tfidf_features/'+str(args.mode)+'_unigram_tfidf.csv',index=False)
		tfidf_vectorizer=TfidfVectorizer(ngram_range=(2,2))
		tfidf_matrix = tfidf_vectorizer.fit_transform(df.iloc[:,2])
		df_bigram=pd.DataFrame(list(tfidf_matrix))
		df_bigram.to_csv('word_tfidf_features/'+str(args.mode)+'_bigram_tfidf.csv',index=False)
		tfidf_vectorizer=TfidfVectorizer(ngram_range=(1,2))
		tfidf_matrix = tfidf_vectorizer.fit_transform(df.iloc[:,2])
		df_uni_bi=pd.DataFrame(list(tfidf_matrix))
		df_uni_bi.to_csv('word_tfidf_features/'+str(args.mode)+'_unigram_bigram_tfidf.csv',index=False)
	if args.features == 'shallow':
		pos_samples = []
		for i in range(df.shape[0]):
			tagged_sentence=pos_tagger(df.ix[i,2])
			tags = [tag[1] for tag in tagged_sentence]
			pos_samples.append(str(tags))
		tfidf_vectorizer=TfidfVectorizer(ngram_range=(1,1))
		tfidf_matrix = tfidf_vectorizer.fit_transform(pos_samples)
		pos_df=pd.DataFrame(list(tfidf_matrix))
		pos_df.to_csv('shallow_tfidf_features/'+str(args.mode)+'_pos_tags_tfidf.csv',index=False)
	if args.features == 'deep':
		df_features=pd.read_csv('deep_syntax_features/'+str(args.mode)+'_deep_syntax_features.csv',header=0)
		lexicalized_production_rules_data = df_features['lexicalized_production_rules'].values
		tfidf_vectorizer=TfidfVectorizer()
		tfidf_matrix=tfidf_vectorizer.fit_transform(lexicalized_production_rules_data)
		df=pd.DataFrame(list(tfidf_matrix))
		df.to_csv('deep_tfidf_features/'+str(args.mode)+'_feature1_tfidf.csv',index=False)
		unlexicalized_production_rules_data=df_features[ 'unlexicalized_production_rules'].values
		tfidf_vectorizer=TfidfVectorizer()
		tfidf_matrix=tfidf_vectorizer.fit_transform(unlexicalized_production_rules_data)
		df=pd.DataFrame(list(tfidf_matrix))
		df.to_csv('deep_tfidf_features/'+str(args.mode)+'_feature2_tfidf.csv',index=False)
		lexicalized_grandparent_comb=df_features['lexicalized_grandparent_comb'].values
		tfidf_vectorizer=TfidfVectorizer()
		tfidf_matrix=tfidf_vectorizer.fit_transform(lexicalized_grandparent_comb)
		df=pd.DataFrame(list(tfidf_matrix))
		df.to_csv('deep_tfidf_features/'+str(args.mode)+'_feature3_tfidf.csv',index=False)
		unlexicalized_grandparent_comb=df_features['unlexicalized_grandparent_comb'].values
		tfidf_vectorizer=TfidfVectorizer()
		df=pd.DataFrame(list(tfidf_matrix))
		df.to_csv('deep_tfidf_features/'+str(args.mode)+'_feature4_tfidf.csv',index=False)
                tfidf_matrix=tfidf_vectorizer.fit_transform(unlexicalized_grandparent_comb)

		
