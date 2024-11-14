#include <iostream>
#include <storm/api/storm.h>
#include "cuddObj.hh"
#include <storm/models/symbolic/Model.h>
#include <storm/models/ModelType.h>
#include <storm-parsers/api/storm-parsers.h>

using namespace std;

// lav function i Cupaal python script, som er wrapper omkring de ting i Jajapy og giver  
//LoadPrism, Instatiate, fit_Parameters, vil vi ikke kalde i CuPaal. Istedet vil vi kalde en enkelt som gør det hele. Til det skal den bruge en path, random initial varialbes/liste over parametre som skal fittes, vores observations.












class CupaalModel
{
public:
    storm::dd::Add<storm::dd::DdType::CUDD, double> transitions;
    std::vector<std::string> labels;
    storm::dd::Bdd<storm::dd::DdType::CUDD> initialStates;
    storm::dd::Bdd<storm::dd::DdType::CUDD> reachableStates;
    storm::dd::Bdd<storm::dd::DdType::CUDD> actions;
    carl::Variables parameters;
    std::set<storm::expressions::Variable> rows;
    std::set<storm::expressions::Variable> cols;
    storm::models::ModelType type;
    std::shared_ptr<storm::dd::DdManager<storm::dd::DdType::CUDD>> manager;
    storm::dd::Bdd<storm::dd::DdType::CUDD> emissions;

    std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD>> parseAndBuildPrism(std::string const &filename)
    {
        const auto program = storm::api::parseProgram(filename);

        constexpr std::vector<std::shared_ptr<storm::logic::Formula const>> formulas;

        std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD>> symbolicmodel = storm::api::buildSymbolicModel<storm::dd::DdType::CUDD, double>(program, formulas);

        return symbolicmodel;
    }

    CupaalModel(const string &filePath)
    {
        std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD>> model = parseAndBuildPrism(filePath);
        transitions = model->getTransitionMatrix();
        labels = model->getLabels();
        initialStates = model->getInitialStates();
        actions = model->getStates(labels[0]);
        type = model->getType();
        reachableStates = model->getReachableStates();
        manager = model->getManagerAsSharedPointer();

        parameters = model->getParameters();
        rows = model->getRowVariables();
        cols = model->getColumnVariables();
    }

    // compute alpha return bdd
    // What are the parameters and what is the output?
    //  As input it should take initial states, transition matrix, lables, training set, emission matrix, CUDD manager,
    storm::dd::Bdd<storm::dd::DdType::CUDD> Alpha(int lengthOfTraces, CupaalModel model)
    {
        storm::dd::Bdd<storm::dd::DdType::CUDD> alpha = initialStates;
        // int lengthOfTraces = model.emissions;

        for (int i = 1; i <= lengthOfTraces; i++)
        {
            storm::dd::Bdd<storm::dd::DdType::CUDD> alphaTemp0 = Cudd_addApply(manager, Cudd_addTimes, emissions, alpha);
            Cudd_Ref(alphaTemp0);
            storm::dd::Bdd<storm::dd::DdType::CUDD> alphaTemp1 = Cudd_addMatrixMultiply(manager, transitions, alphaTemp0, rows, reachableStates);
            Cudd_Ref(alphaTemp1);
            alpha = Cudd_addSwapVariables(manager, alphaTempl, cols, rows, reachableStates);
            Cudd_Ref(alpha);
            Cudd_RecursiveDeref(manager, alphaTemp0);
            Cudd_RecursiveDeref(manager, alphaTemp1);
        }
        return alpha;
    }

    //     compute alpha(model)
    //     {
    //         // make alpha variable

    //      alpha[0] = pi \hadamard (the column that shows the properbilities
    //      for seing the label we are looking for in each state, the vector
    //      should be num_states long)

    //      //transpose transition matrix/add (swap variables)
    //      int i;
    //      for (i = 1; i <= num_obs; i++)
    //      {
    //          temp = T_actions \multiply alpha[i-1]
    //          alpha[i] = temp \hadamard (the column that shows the properbilities
    //      for seing the label we are looking for in each state, the vector
    //      should be num_states long)
    //      }
    //      return alpha;
    //      }

    void Next(std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD>> model, int state)
    {
    }

    std::vector<string> GenerateSet(std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD>> model, int setSize, carl::Variables parameters, string distribution = "", int minSize = 0, bool timed = false)
    {

        // Attempt to
    }

    std::vector<string> Run(std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD>> model, int numberOfSteps, int currentState = -1)
    {

        std::vector<string> output;
        std::tuple<>;
        if (currentState == -1)
        {
            currentState = ResolveRandom(initialStates);
        }

        while ((end(output) - begin(output)) < numberOfSteps)
        {

            output.push_back(symbol);
            currentState = nextState;
        }

        return output;
    }

    double ResolveRandom(storm::dd::Bdd<storm::dd::DdType::CUDD> initialStates)
    {
        std::vector<double> cumSumOfInitialStates = CumSum(initialStates);
        std::vector<int> nonZeroIndicesOfInitialStates = nonzero(cumSumOfInitialStates);

        double randomDouble = rand() / double(RAND_MAX);
        int i = 0;

        while (randomDouble > cumSumOfInitialStates[nonZeroIndicesOfInitialStates[i]])
        {
            i += 1;
        }
        return nonZeroIndicesOfInitialStates[i];
    }

    std::vector<int> Nonzero(const std::vector<double> &v)
    {
        std::vector<int> indices;
        for (int i = 0; i < v.size(); ++i)
        {
            if (v[i] != 0)
            {
                indices.push_back(i);
            }
        }
        return indices;
    }
    std::vector<double> CumSum(const std::vector<double> &values)
    {
        std::vector<double> cumulativeSum(values.size());
        if (values.empty())
            return cumulativeSum;

        cumulativeSum[0] = values[0]; // Start with the first element.
        for (size_t i = 1; i < values.size(); ++i)
        {
            cumulativeSum[i] = cumulativeSum[i - 1] + values[i];
        }
        return cumulativeSum;
    }
};

// https://github.com/Rapfff/jajapy/blob/6a69a79e25f8d6ab08fba46c3de84f3ddc8c9756/jajapy/base/Model.py#L138
// https://github.com/moves-rwth/storm/blob/master/src/storm/models/symbolic/Model.cpp

// // def generateSet(self, set_size: int, param, distribution=None, min_size=None, timed: bool=False) -> Set:
// // 		seq = []
// // 		val = []
// // 		for i in range(set_size):
// // 			if distribution == 'geo':
// // 				curr_size = min_size + int(geometric(param))
// // 			else:
// // 				if type(param) == list:
// // 					curr_size = param[i]
// // 				elif type(param) == int:
// // 					curr_size = param

// // 			if timed:
// // 				trace = self.run(curr_size,timed)
// // 			else:
// // 				trace = self.run(curr_size)

// // 			if not trace in seq:
// // 				seq.append(trace)
// // 				val.append(1)
// // 			else:
// // 				val[seq.index(trace)] += 1

// // 		return Set(seq,val)

// // def run(self,number_steps: int, timed: bool = False) -> list:
// // 		"""
// // 		Simulates a run of length ``number_steps`` of the model and return the
// // 		sequence of observations generated. If ``timed`` it returns a list of
// // 		pairs waiting time-observation.

// // 		Parameters
// // 		----------
// // 		number_steps: int
// // 			length of the simulation.
// // 		timed: bool, optional
// // 			Wether or not it returns also the waiting times. Default is False.

// // 		Returns
// // 		-------
// // 		output: list of str
// // 			trace generated by the run.
// // 		"""
// // 		output = []
// // 		current = resolveRandom(self.initial_state)
// // 		c = 0
// // 		while c < number_steps:
// // 			[symbol, next_state, time_spent] = self.next(current)
// // 			output.append(symbol)
// // 			if timed:
// // 				output.append(time_spent)
// // 			current = next_state
// // 			c += 1
// // 		output.append(self.labelling[current])
// // 		return output

// // def resolveRandom(m: list) -> int:
// //      m = array(m).cumsum()
// // 	    mi = nonzero(m)[0]
// // 	    r = random()
// // 	    i = 0
// // 	    while r > m[mi[i]]:
// // 		    i += 1

// // 	    return mi[i]

// // 		----------// 		----------// 		----------// 		----------// 		----------// 		----------// 		----------

// // 		"""
// // 		Generates a set (training set / test set) containing ``set_size`` traces.

// // 		Parameters
// // 		----------
// // 		set_size: int
// // 			number of traces in the output set.
// // 		param: a list, an int or a float.
// // 			the parameter(s) for the distribution. See "distribution".
// // 		distribution: str, optional
// // 			If ``distribution=='geo'`` then the sequence length will be
// // 			distributed by a geometric law such that the expected length is
// // 			``min_size+(1/param)``.
// // 			If distribution==None param can be an int, in this case all the
// // 			seq will have the same length (``param``), or ``param`` can be a
// // 			list of int.
// // 			Default is None.
// // 		min_size: int, optional
// // 			see "distribution". Default is None.
// // 		timed: bool, optional
// // 			Only for timed model. Generate timed or non-timed traces.
// // 			Default is False.

// // 		Returns
// // 		-------
// // 		output: Set
// // 			a set (training set / test set).

// // 		Examples
// // 		--------
// // 		>>> set1 = model.generateSet(100,10)
// // 		>>> # set1 contains 100 traces of length 10
// // 		>>> set2 = model.generate(100, 1/4, "geo", min_size=6)
// // 		>>> # set2 contains 100 traces. The length of the traces is distributed following
// // 		>>> # a geometric distribution with parameter 1/4. All the traces contains at
// // 		>>> # least 6 observations, hence the average length of a trace is 6+(1/4)**(-1) = 10.
// // 		"""