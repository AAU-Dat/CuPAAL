#include <iostream>
#include <storm/api/storm.h>
#include "src/cupaal/baum.h"
#include "cuddObj.hh"
#include <storm/models/symbolic/Model.h>
#include <storm/models/ModelType.h>

using namespace std;

class CupaalModel
{
public:
    storm::dd::Add<storm::dd::DdType::CUDD, double> transitions;
    std::vector<std::string> labels;
    storm::dd::Bdd<storm::dd::DdType::CUDD> initialStates;
    storm::dd::Bdd<storm::dd::DdType::CUDD> actions;
    carl::Variables parameters;
    std::set<storm::expressions::Variable> rows;
    std::set<storm::expressions::Variable> cols;
    storm::models::ModelType type;

    CupaalModel(const string &filePath)
    {
        std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD>> model = parser::parseAndBuildPrism(filePath);
        transitions = model->getTransitionMatrix();
        labels = model->getLabels();
        initialStates = model->getInitialStates();
        actions = model->getStates(labels[0]);
        parameters = model->getParameters();
        rows = model->getRowVariables();
        cols = model->getColumnVariables();
        type = model->getType();
    }

    GenerateSet(){
        
    }
};



// def generateSet(self, set_size: int, param, distribution=None, min_size=None, timed: bool=False) -> Set:
// 		"""
// 		Generates a set (training set / test set) containing ``set_size`` traces.

// 		Parameters
// 		----------
// 		set_size: int
// 			number of traces in the output set.
// 		param: a list, an int or a float.
// 			the parameter(s) for the distribution. See "distribution".
// 		distribution: str, optional
// 			If ``distribution=='geo'`` then the sequence length will be
// 			distributed by a geometric law such that the expected length is
// 			``min_size+(1/param)``.
// 			If distribution==None param can be an int, in this case all the
// 			seq will have the same length (``param``), or ``param`` can be a
// 			list of int.
// 			Default is None.
// 		min_size: int, optional
// 			see "distribution". Default is None.
// 		timed: bool, optional
// 			Only for timed model. Generate timed or non-timed traces.
// 			Default is False.
		
// 		Returns
// 		-------
// 		output: Set
// 			a set (training set / test set).
		
// 		Examples
// 		--------
// 		>>> set1 = model.generateSet(100,10)
// 		>>> # set1 contains 100 traces of length 10
// 		>>> set2 = model.generate(100, 1/4, "geo", min_size=6)
// 		>>> # set2 contains 100 traces. The length of the traces is distributed following
// 		>>> # a geometric distribution with parameter 1/4. All the traces contains at
// 		>>> # least 6 observations, hence the average length of a trace is 6+(1/4)**(-1) = 10.
// 		"""
// 		seq = []
// 		val = []
// 		for i in range(set_size):
// 			if distribution == 'geo':
// 				curr_size = min_size + int(geometric(param))
// 			else:
// 				if type(param) == list:
// 					curr_size = param[i]
// 				elif type(param) == int:
// 					curr_size = param

// 			if timed:
// 				trace = self.run(curr_size,timed)
// 			else:
// 				trace = self.run(curr_size)

// 			if not trace in seq:
// 				seq.append(trace)
// 				val.append(1)
// 			else:
// 				val[seq.index(trace)] += 1

// 		return Set(seq,val)