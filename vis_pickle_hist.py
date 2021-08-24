#!/usr/bin/python

import os,sys,pickle,argparse,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from scipy import stats
from sklearn import preprocessing
from scipy.stats import mstats
import matplotlib


#matplotlib.rc('font', **{'family' : "sans-serif"})
#params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description = "Used for testing models")
parser.add_argument('-f',default='/home/mleming/Desktop/MGH_ML_pipeline/pandas/cache/sample.pkl',type=str)
parser.add_argument('-t',default='Test_Control',type=str)
parser.add_argument('-c',default='SexDSC',type=str)
parser.add_argument('-b',default=False,action='store_true')
parser.add_argument('-n',action='store_true',default=False)
parser.add_argument('-ct',default=-1,type=int)
parser.add_argument('--title',default="",type=str)
parser.add_argument('--dpi',default=900,type=int)
parser.add_argument('--output',default='',type=str)
parser.add_argument('--kruskal',default=False,action='store_true')
parser.add_argument('--csv_output',default=False,action='store_true')
parser.add_argument('--latex',default = False,action='store_true')
parser.add_argument('--bold_var',default = "",type=str)
parser.add_argument('--log_color_scale',default=False,action='store_true')
parser.add_argument('--colormap',type=str,default="viridis")
args = parser.parse_args()
if args.kruskal:
	matplotlib.rc('text', usetex = True)	
	matplotlib.rc('axes', linewidth=2)
	matplotlib.rc('font', weight='bold')
	matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath \usepackage{xcolor}']

def is_float(n):
	try:
		float(n)
		return True
	except:
		return False

def get_sorted(confounds):
	c_unique = list(set(confounds))
	if None in c_unique:
		c_unique.remove(None)
	if np.all([is_float(i) for i in c_unique]):
		c_unique = [str(k) for k in sorted([int(i) for i in c_unique])]
	else:
		c_unique = sorted(c_unique)
	e_ind = -1
	for i in range(len(c_unique)):
		if c_unique[i].lower() == "exclude":
			e_ind = i
	if e_ind != -1:
		foo = c_unique[e_ind]
		del c_unique[e_ind]
		c_unique.append(foo)
	return c_unique

def iz_nan(k,inc_null_str=False):
	if k is None:
		return True
	if inc_null_str and isinstance(k,str):
		if k.lower() == "null" or k.lower() == "unknown":
			return True
	try:
		if np.isnan(k):
			return True
		else:
			return False
	except:
		if k == np.nan:
			return True
		else:
			return False


if args.output != "":
	import matplotlib
	matplotlib.use('Agg')
	if not os.path.isdir(os.path.dirname(args.output)):
		os.makedirs(os.path.dirname(args.output))


df = pd.read_pickle(args.f)

working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
tech_covars_file = os.path.join(working_dir,'json','technical.txt')
tech_covars = {}
if os.path.isfile(tech_covars_file):
	with open(tech_covars_file,'r') as fileobj:
		for line in fileobj.readlines():
			line = line.replace('\n','')
			if line[0] == '/':
				tech_covars[line[1:]] = "demo"
			elif line[0] == '\\':
				tech_covars[line[1:]] = "tech"

df.drop(df.loc[df['Ages'] < 0].index,inplace=True)

xlab = args.c
test_vars = df[args.t].to_numpy()
confounds = df[args.c].to_numpy()
title = os.path.splitext(os.path.basename(args.f))[0].replace("_"," ").capitalize()
if args.kruskal:
	title = "%s variable correlations (n = %d)"%(title,len(test_vars))


c_discrete = isinstance(confounds[0],str)
t_discrete = isinstance(test_vars[0],str)

if args.latex:
	if t_discrete:
		unique_vars = get_sorted(np.unique(list(filter(lambda k: not iz_nan(k),test_vars))))
		total_str = "TOTAL"
		nan_str = "% MISSING"
		df_table = pd.DataFrame(columns = sorted(df.columns),index=unique_vars + [total_str,nan_str])
		categorical = np.zeros((len(df.columns),))
		for i in range(len(df.columns)):
			c = df.columns[i]
			json_label_file = os.path.join(working_dir,'json','labels','%s.json' % c)
			if os.path.isfile(json_label_file):
				jj = json.load(open(json_label_file,'r'))
				if jj["discrete"] or c == "MRN":
					categorical[i] = 1
				else:
					categorical[i] = 2
		for i in range(len(df.columns)):
			c = df.columns[i]
			for v in unique_vars + [total_str,nan_str]:
				if v == total_str or v == nan_str:
					select = np.ones((len(df[args.t]),)).astype(bool)
				else:
					select = df[args.t] == v
				foo = df[c].loc[select].to_numpy()
				len_all = len(foo)
				foo = list(filter(lambda k: not iz_nan(k,inc_null_str=True),foo))
				len_nonnan = len(foo)
				if len_all > 0:
					perc_nonnan = len_nonnan / float(len_all)
				else:
					perc_nonnan = 1
				if v == nan_str:
					df_table.at[v,c] = "%.2f" % ((1 - perc_nonnan) * 100)
				elif len(foo) == 0:
					df_table.at[v,c] = 0
				elif categorical[i] == 1:
					df_table.at[v,c] = len(np.unique(foo))
				elif categorical[i] == 2:
					df_table.at[v,c] = "%.2f ± %.2f" % (np.mean(foo),np.std(foo))
		for t in tech_covars:
			if tech_covars[t] == "tech":	
				t2 = "%s*" % t
			elif tech_covars[t] == "demo":
				t2 = "%s✝" % t
			else:
				t2 = t
			df_table.rename(columns = {t:t2},inplace=True)
			df.rename(columns = {t:t2},inplace=True)
		if args.output == "":
			print(df_table.T.to_latex())
		else:
			egg = False
			temp = df_table[sorted(df.columns[categorical == 1])]
			if egg:
				egg_var = np.array(list(map(lambda k: float(k) < 99,temp.T[nan_str].to_numpy())))
				temp = temp.T[egg_var]
			else:
				temp = temp.T
			temp.to_latex(buf=open(args.output.replace('.','_categorical.'),'w'))
			
			temp = df_table[sorted(df.columns[categorical == 2])]
			if egg:
				egg_var = np.array(list(map(lambda k: float(k) < 99,temp.T[nan_str].to_numpy())))
				temp = temp.T[egg_var]
			else:
				temp = temp.T

			temp.to_latex(buf=open(args.output.replace('.','_continuous.'),'w'))
			
			
			
			#df_table[sorted(df.columns[np.logical_and(categorical == 1,egg_var)])].T.to_latex(buf=open(args.output.replace('.','_categorical.'),'w'))
			#df_table[sorted(df.columns[np.logical_and(categorical == 2,egg_var)])].T.to_latex(buf=open(args.output.replace('.','_continuous.'),'w'))
	exit()

fig, axs = plt.subplots(1, 1, tight_layout=True)
plt.title(title if args.title == "" else args.title)

def is_discrete(c):
	return c.dtype == object
	i=0
	try:
		while iz_nan(c[i]):
			i += 1
		return isinstance(c[i],str)
	except:
		return True

def handle_null_single(c1,remove=False):
	nanlist = np.array(list(map(lambda k: iz_nan(k),c1)))
	if remove:
		c1 = c1[~nanlist]
	else:
		if is_discrete(c1):
			c1[nanlist] = "NULL"
		else:
			raise Exception("Cannot replace nan values for non-categorical variable")
	return c1

def handle_null_double(c1,c2):
	c1_discrete = c1["discrete"]
	c2_discrete = c2["discrete"]
	c1_nanlist = c1["nanlist"] #np.array(list(map(lambda k: iz_nan(k),c1)))
	c2_nanlist = c2["nanlist"] #np.array(list(map(lambda k: iz_nan(k),c2)))
	#final_nanlist = np.logical_or(c1_nanlist,c2_nanlist)
	#if c1_discrete:
	#	c1[c1_nanlist] = "NULL"
	#if c2_discrete:
	#	c2[c2_nanlist] = "NULL"
	if c1_discrete and c2_discrete:
		return c1,c2
	elif c1_discrete and not c2_discrete:
		final_nanlist = c2_nanlist
	elif not c1_discrete and c2_discrete:
		final_nanlist = c1_nanlist
	elif not c1_discrete and not c2_discrete:
		final_nanlist = np.logical_or(c1_nanlist,c2_nanlist)
	else:
		raise Exception("Boolean logic is wrong")
	return c1[~final_nanlist],c2[~final_nanlist]

def compare_columns(c1,c2):
	#c1,c2 = handle_null_double(c1,c2)
	c1_discrete = c1["discrete"]
	c2_discrete = c2["discrete"]
	if c1_discrete and c2_discrete and False:
		#chi-squared test
		if "bool" in c1 and "bool" in c2:
			c1_unique = c1["unique"]
			c2_unique = c2["unique"]
			observed = np.zeros((len(c1_unique),len(c2_unique)))
			expected = np.zeros(observed.shape)
			c1_bool = c1["bool"]
			c2_bool = c2["bool"]
			observed[:,:] = np.dot(c1_bool,c2_bool.T)
			expected[:,:] = np.dot(c1["boolmean"],c2["boolsum"].T)
#			for i in range(observed.shape[0]):
#				c1_bool = 
#				for j in range(observed.shape[1]):
#					c2_bool = c2 == c2_unique[j]
#					observed[i,j] = np.sum(np.logical_and(c1_bool,c2_bool))
#					expected[i,j] = np.sum(np.mean(c1_bool) * np.mean(c2_bool) * len(c1))
			chisq,p1 = stats.chisquare(observed,expected,axis=0)
			p1 = np.expand_dims(p1,axis=1)
			#print("--")
			#print(p1.shape)
			#print(c2["boolmean"].shape)
			#print(c1["boolmean"].shape)
			#print((p1 * c2["boolmean"]).shape)
			p=np.sum(p1 * c2["boolmean"])
			#chisq,p2 = stats.chisquare(observed,expected,axis=0)
			#p2=min(p2)
			#p = min(p1,p2)
		else:
			return np.nan
	elif not c1_discrete and not c2_discrete:
		final_nanlist = np.logical_or(c1["nanlist"],c2["nanlist"])
		try:
			foo1 = c1["np"][~final_nanlist]
			foo2 = c2["np"][~final_nanlist]
			if np.all(foo1 == foo2):
				p = 1
			else:
				s,p = stats.mannwhitneyu(foo1,foo2)
		except:
			p = 0
		assert(not(iz_nan(p)))
	elif c1_discrete:# and not c2_discrete:
		if "bool" in c1:
			c1_unique = c1["unique"]
			c1_noc2null = c1["np"][~c2["nanlist"]]
			t = [c2["excludenull"][c1_noc2null == c] for c in c1_unique]
			t = list(filter(lambda k: len(k) > 30,t))
			
			if len(t) < 2:
				return 1
			if np.all([len(np.unique(k)) == 1 for k in t]):
				return 1
			#s,p = stats.f_oneway(*t)
			s,p = stats.kruskal(*t)
			assert(not(iz_nan(p)))
			return p
		else:
			return np.nan
	elif not c1_discrete and c2_discrete:
		return np.nan
		return compare_columns(c2,c1)
	else:
		raise Exception("This line shouldn't be possible")
	return p

working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
var_list_file = os.path.join(working_dir,'txt','include_vars.txt')

if args.kruskal:
	binarize = args.b
	dlist = []
	if os.path.isfile(var_list_file):
		print("Loading var list")
		with open(var_list_file,'r') as fileobj:
			columns = []
			for row in fileobj.readlines():
				columns.append(row[:-1])
	else:
		columns = sorted(df.columns)
	for c in columns:
		c1 = df[c].to_numpy()
		temp = {"np" : c1,"nanlist" : np.array(list(map(lambda k: iz_nan(k),c1))),"discrete":is_discrete(c1)}
		if temp["discrete"]:
			temp["np"][temp["nanlist"]] = "NULL"
			temp["unique"] = np.unique(temp["np"])
			if len(temp["unique"]) < 10000 and len(temp["unique"]) > 1:
				temp["bool"] = np.zeros((len(temp["unique"]),len(c1)))
				for i in range(temp["bool"].shape[0]):
					temp["bool"][i,:] = (c1 == temp["unique"][i]).astype(int)
				temp["boolsum"]  = np.expand_dims(temp["bool"].sum(axis=1),axis=1)
				temp["boolmean"] = np.expand_dims(temp["bool"].mean(axis=1),axis=1)
			elif len(temp["unique"]) >= 100:
				continue
			temp["nonull"] = c1.copy()
			temp["nonull"][temp["nanlist"]] = "NULL"
		temp["excludenull"] = (c1.copy())[~temp["nanlist"]]
		temp["label"] = c
		dlist.append(temp)
	dlist = list(filter(lambda k: ("bool" in k or not k["discrete"]),dlist))
	labels = [d["label"] for d in dlist]
	krusks = np.zeros((len(dlist),len(dlist)))
	for i in range(krusks.shape[0]):
		print(i)
		for j in range(krusks.shape[1]):
			if i == j:
				continue
			c1 = dlist[i]
			c2 = dlist[j]
			p = compare_columns(c1,c2)
			#if np.isnan(p):
			#	p = 1
			#krusks[j,i] = 1 if (p > 0.05) else 0 # Read along the columns
			if binarize: p = 1 if (p > 0.05) else 0
			krusks[i,j] = p # Read along the rows
			#krusks[j,i] = p
	krusks = krusks.T
	if args.log_color_scale:
		krusks = krusks + 1e-52#+ np.nextafter(0,1) # Finding log of zero is a no-no
	else:
		krusks[krusks > 0.1] = 0.1
	plt.imshow(krusks,norm= None if not args.log_color_scale else matplotlib.colors.LogNorm(),cmap=plt.get_cmap(args.colormap))
	
	if not args.b:	
		if args.log_color_scale:
			cbar = plt.colorbar(fraction=0.046, pad=0.04)
			cbar.ax.set_ylabel('p-value (log scale)', rotation=270,labelpad=10.0)
		else:
			cbar = plt.colorbar(ticks=[0,0.05,0.1],fraction=0.046, pad=0.04)
			cbar.ax.set_ylabel('p-value', rotation=270,labelpad=10.0)
			cbar.ax.set_yticklabels(['$0$', '$0.05$', '$>0.1$'])
	for l in labels:
		print(l)
	plt.xticks(list(range(len(labels))),[str(c) for c in labels],fontsize=6,rotation=90)
	boldstr = r'\textcolor{red}{\textbf{%s}}'
	plt.yticks(list(range(len(labels))),[str(labels[i]) if not (args.bold_var in labels) else str(labels[i]) if not krusks[i,labels.index(args.bold_var)] <= 0.05 else boldstr  % str(labels[i]) for i in range(len(labels))],fontsize=6)
	plt.ylabel('Continuous \hspace*{5em} Categorical \hspace*{2em}',fontsize=24)
	plt.xlabel('Categorical \hspace*{5em} Continuous \hspace*{2em}',fontsize=24)
	fig = plt.gcf()
	fig.set_size_inches(10.5, 10.5)
	if args.output != "":
		plt.savefig(args.output, dpi=args.dpi)
	else:
		plt.show()
	exit()

#if args.csv_output:
#	continue

if args.n:
	plt.yticks([])

if t_discrete:
	#t_unique = np.unique(test_vars)
	t_unique = get_sorted(test_vars)
	alpha=0.5
	if c_discrete:
		alpha=1
		c_unique = get_sorted(confounds)
		if args.ct != -1:
			args.ct = min(args.ct,len(c_unique))
			if args.ct != len(c_unique):
				xlab = "%s (top %d)" % (xlab, args.ct)
			foo = [len(confounds[confounds == c]) for c in c_unique]
			foos = (sorted(foo)[::-1])[args.ct-1]
			ll = []
			for i in range(len(foo)):
				if foo[i] < foos:
					ll.append(c_unique[i])
			for l in ll:
				c_unique.remove(l)
		X = np.arange(len(c_unique))
		w = (0.75 / len(t_unique))
		plt.xticks(X + w/2.0,c_unique,rotation = 45)
	else:
		hist,bin_edges = np.histogram(confounds[list(map(lambda k: k is not None and not np.isnan(k),confounds))],bins=100)
	for i in range(len(t_unique)):
		t = t_unique[i]
		cv = confounds[test_vars == t]
		lt = "%s (n = %d)" % (t.replace("_"," ").upper(),len(cv))
		if c_discrete:
			cv_counts = [len(cv[cv == c]) for c in c_unique]
			axs.bar(X + i * w,cv_counts,width = w,label=lt)
		else:
			axs.hist(cv, bins=bin_edges,label=lt,alpha=alpha,density=args.n)
	plt.legend(loc='upper right')
else:
	if c_discrete:
		print("Can't really visualize this sort of thing")
	else:
		plt.scatter(confounds,test_vars,alpha=0.5)
		plt.ylabel(args.t)

plt.xlabel(xlab)
if args.output != "":
	plt.savefig(args.output,dpi=args.dpi)
else:
	plt.show()

