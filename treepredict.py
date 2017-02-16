my_data=[['slashdot','USA','yes',18,'None'],
		 ['google','France','yes',23,'Premium'],
		 ['digg','USA','yes',24,'Basic'],
		 ['kiwitobes','France','yes',23,'Basic'],
		 ['google','UK','no',21,'Premium'],
		 ['(direct)','New Zealand','no',12,'None'],
		 ['(direct)','UK','no',21,'Basic'],
		 ['google','USA','no',24,'Premium'],
		 ['slashdot','France','yes',19,'None'],
		 ['digg','USA','no',18,'None'],
		 ['google','UK','no',18,'None'],
		 ['kiwitobes','UK','no',19,'None'],
		 ['digg','New Zealand','yes',12,'Basic'],
		 ['slashdot','UK','no',21,'None'],
		 ['google','UK','yes',18,'Basic'],
		 ['kiwitobes','France','yes',19,'Basic']]

# Divides a set on a specific column. Can handle numeric or nominal values
def divideset(rows, column, value):
	split_function = None
	
	if isinstance(value, int) or isinstance(value, float):
			split_function = lambda row:row[column] >= value
	else:
			split_function = lambda row:row[column] == value

	set1 = [row for row in rows if split_function(row)]
	set2 = [row for row in rows if not split_function(row)]
	return (set1,set2)

# Create counts of possible results (the last column of each row is the result)
def uniquecounts(rows):
		results = {}
		for row in rows:
				y = row[len(row)-1]
				if y not in results: results[y]=0
				results[y]+=1
		return results

# Entropy is the sum of pxlog(px) across all the different possible results
def entropy(rows):
		from math import log
		log2 = lambda x:log(x)/log(2)
		results = uniquecounts(rows)
		ent = 0.0
		for r in results:
			p = float(results[r])/len(rows)
			ent = ent - p*log2(p)
		return ent

# train and build the decision tree
def buildtree(rows, scoref=entropy):
	if len(rows)==0: return decisionnode()
	current_score = scoref(rows)
		
	best_gain = 0.0
	best_criteria = None
	best_sets = None

	column_count = len(rows[0])-1

	# iterate each col-value pair, use the value to calculate entropy and get the best gains
	for col in range(0,column_count):
			column_values={}
			
			for row in rows:
					column_values[row[col]]=1

			for value in column_values.keys():
					(set1,set2) = divideset(rows,col,value)
					p = float(len(set1))/len(rows)
					gain = current_score - p*scoref(set1) - (1-p)*scoref(set2)

					if gain > best_gain and len(set1)>0 and len(set2)>0:
							best_gain = gain
							best_criteria = (col, value)
							best_sets = (set1,set2)

	if best_gain > 0:
			trueBranch = buildtree(best_sets[0])
			falseBranch = buildtree(best_sets[1])
			return decisionnode(col=best_criteria[0], value = best_criteria[1], tb=trueBranch, fb=falseBranch)
	else:
		return decisionnode(results=uniquecounts(rows))


def printtree(tree, indent=''):
		if tree.results != None:
				print str(tree.results)
		else:
				print str(tree.col) + ':' + str(tree.value) + '? '

				print indent + 'T-> ',
				printtree(tree.tb, indent+'  ')
				print indent + 'F-> ',
				printtree(tree.fb, indent+'  ')

def classify(observation, tree):
		if tree.results!=None:
				counts = uniquecounts(my_data)
				results = {}
				for k,v in tree.results.items():
						results[k] = float(v)/counts[k]
				return results
		else:
				v = observation[tree.col]
				branch = None
				if isinstance(v,int) or isinstance(v,float):
						if v >= tree.value: branch = tree.tb
						else: branch = tree.fb
				else:
						if v == tree.value: branch = tree.tb
						else: branch = tree.fb
				return classify(observation, branch)

# to avoid overfitting, we need prune the tree, an overfitting tree may give
# an answer as being more certain than it really is by creating branches that 
# decrease entropy slightly for the training set
# the algorithm to prune is to checking pairs of nodes that have a common parent
# to see if merging them would increase the entropy by less than a specified threshold, if so
# we did the merge.
def prune(tree, mingain):
	if tree.tb.results == None:
			prune(tree.tb, mingain)
	if tree.fb.results == None:
			prune(tree.fb, mingain)
	
	if tree.tb.results != None and tree.fb.results != None:
			tb,fb = [],[]
		    # because entropy needs exact copies of the result column	
			for v,c in tree.tb.results.items():
					tb+=[[v]]*c
			for v,c in tree.fb.results.items():
					fb+=[[v]]*c

			delta = entropy(tb+fb) - (entropy(tb) + entropy(fb)/2)

			if delta < mingain:
					tree.tb, tree.fb = None, None
					tree.results = uniquecounts(tb+fb)

# if you are missing data,  you can actually follow both branches. However,
# instead of counting the results equally, the results from either side are weighted.
def mdclassify(observation, tree):
		if tree.results != None:
				counts = uniquecounts(my_data)
				results = {}
				for k,v in tree.results.items():
						results[k] = float(v)/counts[k]
				return results
		else:
				v=observation[tree.col]

				if v == None:
						tr,fr = mdclassify(observation, tree.tb),mdclassify(observation, tree.fb)
						tcount = sum(tr.values())
						fcount = sum(fr.values())
						tw = float(tcount)/(tcount+fcount)
						fw = float(fcount)/(tcount+fcount)
						result={}
						for k,v in tr.items(): result[k]=v*tw
						for k,v in fr.items(): result[k]=v*fw
						return result
				else:
					if isinstance(v,int) or isinstance(v,float):
						if v >= tree.value: branch = tree.tb
						else: branch = tree.fb
					else:
						if v == tree.value: branch = tree.tb
						else: branch = tree.fb				
					return mdclassify(observation,branch)


#use CART algorithm to train decision tree
class decisionnode:
		def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
				self.col=col # col is the column index of the criteria to be tested
				self.value=value	# value is the value that the column must match to get a true result
				self.results=results # result stores a dictionary of results for this branch. This is None for everything except endpoint
				self.tb=tb # tb and fb are decisionnodes, which are the next nodes in the tree if the results is true or false, respectively
				self.fb=fb
				

		
