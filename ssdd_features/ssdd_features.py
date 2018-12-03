from pattern.en import tag as pos_tagger
from nltk.tokenize import RegexpTokenizer
from nltk import nonterminals, Nonterminal, Production, CFG
from BerkeleyParser import parser
from nltk.tree import Tree, _child_names
from nltk.util import ngrams
import nltk
import argparse
import os
import pandas as pd
import csv
import string

jar_path = 'berkeleyparser/BerkeleyParser-1.7.jar'
gr_path = 'berkeleyparser/eng_sm6.gr'

bp = parser(jar_path,gr_path)



tokenizer = RegexpTokenizer(r'\w+')

# Features from syntactic stylometry for deception detection
# Feature encoding -- 
# unigram, bigram , union
# shallow syntax -- POS tags, POS tags+unigram features
# deep syntax -- 
# r - unlexicalized production rules (all production rules except those with terminal nodes)
# r_star: lexicalized production rules( all production rules)
# r_cap: unlexicalized production rules combined with the grandparent node 
# r_sc: lexicalized prod rules combined with the grandparent node



parser=argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str,help="directory containing files")
parser.add_argument('--features',type=str,choices=['words','shallow','deep'],help="features to generate")
parser.add_argument('--mode',type=str,choices=['train','valid','test'],help="file to parse")
args=parser.parse_args()

def parse_sentence(train_news,i):
	row = train_news.ix[i,2]
	return bp.parse(row)


def generate_features(train_news,args):
	if 'words' in args.features:
		# extract unigram, bigram, union of the two
		# train news --> [id label statement]
		with open('word_features/'+str(args.mode)+"_word_features.csv","a") as f:
			writer = csv.writer(f)
			writer.writerow(["unigram","bigram"])
			for i in range(train_news.shape[0]):
				sample = train_news.ix[i,2]
				train_unigrams = list(nltk.ngrams(sample.split(" "),1))
				train_bigrams=list(nltk.bigrams(sample.split(" ")))
				writer.writerow([train_unigrams, train_bigrams])	
	
	if 'shallow' in args.features:
		# POS tags
		with open('shallow_syntax_features/'+str(args.mode) + "_shallow_syntax_features.csv","a") as f:
			writer=csv.writer(f)
			writer.writerow(["pos tags"])
			for i in range(train_news.shape[0]):
				pos_tags=pos_tagger(train_news.ix[i,2])
				writer.writerow(pos_tags)	

	if 'deep' in args.features:
		# r - unlexicalized production rules (all production rules except those with terminal nodes)
		# r_star: lexicalized production rules( all production rules)
		# r_cap: unlexicalized production rules combined with the grandparent node 
		# r_sc: lexicalized prod rules combined with the grandparent node
		with open('deep_syntax_features/'+str(args.mode) + "_deep_syntax_features.csv","a") as f:
			writer=csv.writer(f)
			writer.writerow(["lexicalized_production_rules","unlexicalized_production_rules","lexicalized_grandparent_comb","unlexicalized_grandparent_comb"])
			for i in range(train_news.shape[0]):
				print(train_news.ix[i,2])
				tree = parse_sentence(train_news,i)
				tree = nltk.Tree.fromstring(tree)
				# lexicalized production rules --> all rules
				rules = tree.productions()
				# unlexialized production rules --> no terminal nodes
				rules_r = get_unlexicalized(rules)
				# lexicalized production rules combined with grandparent node
				rules_r_sc=lexicalized_grandparent(tree)
				# unlexicalized production rules combined with grandparent node
				rules_r_cap=unlexicalized_grandparent(rules_r_sc)
				writer.writerow([rules, rules_r, rules_r_sc, rules_r_cap])
			
			
def unlexicalized_grandparent(lexicalized_grandparent_rules):
	return [rule for rule in lexicalized_grandparent_rules if rule.is_nonlexical()]

def get_unlexicalized(rules):
	return [rule for rule in rules if rule.is_nonlexical()]
						
def lexicalized_grandparent(tree):
	productions_ = [Production(Nonterminal(tree._label),_child_names(tree))]
	for child in tree:
		if isinstance(child,Tree):
			productions_+=productions(child,tree._label)
	return productions_			

def productions(tree, parent):
	prods = [Production(Nonterminal(parent + "^" + tree._label), _child_names(tree))]
	for child in tree:
		if isinstance(child,Tree):
			prods += productions(child, tree._label)
	return prods

def get_data(args):
	if args.mode == 'train':
		filename=os.path.join(args.data_dir,"train.tsv")
	elif args.mode == 'test':
		filename=os.path.join(args.data_dir,"test.tsv")
	elif args.mode=='valid':
		filename=os.path.join(args.data_dir,"valid.tsv")
	df=pd.read_csv(filename, delimiter='\t',header=None)
	df=df.iloc[:,0:3]
	print(df.head())
	return df

if __name__ == '__main__':
	data = get_data(args)
	features = generate_features(data,args)
