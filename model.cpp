#include <iostream>
#include <storm/api/storm.h>
#include "cuddObj.hh"
#include <storm/models/symbolic/Model.h>
#include <storm/models/ModelType.h>
#include <storm-parsers/api/storm-parsers.h>

using namespace std;

class CupaalModel
{
public:
    DdManager* manager;
    DdNode* transitions;                                //P
    storm::dd::Bdd<storm::dd::DdType::CUDD> emissions;  //Omega
    DdNode* initialStates;                              //PI
    DdNode* reachableStates;                            //All States


    // vector<string> labels;
    // storm::dd::Bdd<storm::dd::DdType::CUDD> actions;
    // carl::Variables parameters;
    // DdNode** rowVars;
    // DdNode** colVars;
    // storm::models::ModelType type;
    // ssize_t numberOfStates; 

    std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD>> parseAndBuildPrism(std::string const &filename)
    {
        const auto program = storm::api::parseProgram(filename);

        const std::vector<std::shared_ptr<storm::logic::Formula const>> formulas;

        std::shared_ptr<storm::models::symbolic::Model<storm::dd::DdType::CUDD>> symbolicmodel = storm::api::buildSymbolicModel<storm::dd::DdType::CUDD, double>(program, formulas);

        return symbolicmodel;
    }

    CupaalModel(const string &filePath)
    {
        std::shared_ptr<storm::models::Model<double>> model = parseAndBuildPrism(filePath);
        manager = model->getManager().getInternalDdManager().getCuddManager().getManager();
        
        transitions = model->getTransitionMatrix().getInternalAdd().getCuddDdNode();
        initialStates = model->getInitialStates().getInternalBdd().getCuddDdNode();
        reachableStates = model->getReachableStates().getInternalBdd().getCuddDdNode(); //Number of states
        emissions = model->;
    }

    void BaumWelch()
    {
        // Compute alpha
        // Compute beta

        // Compute gamma
        // Compute xi

        // Update initial states based gamma
        // Transitions are updated based on gamma and xi
        // Emmission matrix is updated based on gamma + observations and gamma
    }

    //From forwards_backwards.c in Sudd development branch
    DdNode **_forwards(DdNode **omega, DdNode *pi, DdNode **row_vars, DdNode **column_vars, int n_vars, int n_obs)
    {
        DdNode **alpha = (DdNode**) malloc(sizeof(n_obs + 1));
        alpha[0] = pi;

        for (int t = 1; t <= n_obs; t++)
        {
            DdNode *alpha_temp_0 = Cudd_addApply(manager, Cudd_addTimes, omega[t - 1], alpha[t - 1]);
            Cudd_Ref(alpha_temp_0);
            DdNode *alpha_temp_1 = Cudd_addMatrixMultiply(manager, transitions, alpha_temp_0, row_vars, n_vars);
            Cudd_Ref(alpha_temp_1);
            alpha[t] = Cudd_addSwapVariables(manager, alpha_temp_1, column_vars, row_vars, n_vars);
            Cudd_Ref(alpha[t]);
            Cudd_RecursiveDeref(manager, alpha_temp_0);
            Cudd_RecursiveDeref(manager, alpha_temp_1);
        }

        return alpha;
    }

    DdNode** _backwards(DdNode **omega, DdNode *pi, DdNode **row_vars, DdNode **column_vars, int n_vars, int n_obs){
    DdNode* _P = Cudd_addSwapVariables(manager, transitions, column_vars, row_vars, n_vars);
    Cudd_Ref(_P);

    DdNode** beta = (DdNode**) malloc(sizeof(n_obs + 1));
    beta[n_obs] = Cudd_ReadOne(manager);

    for (int t = n_obs - 1; 0 <= t; t--) {
        DdNode* beta_temp_0 = Cudd_addMatrixMultiply(manager, _P, beta[t + 1], row_vars, n_vars);
        Cudd_Ref(beta_temp_0);
        DdNode* beta_temp_1 = Cudd_addSwapVariables(manager, beta_temp_0, column_vars, row_vars, n_vars);
        Cudd_Ref(beta_temp_1);
        beta[t] = Cudd_addApply(manager, Cudd_addTimes, omega[t], beta_temp_1);
        Cudd_Ref(beta[t]);
        Cudd_RecursiveDeref(manager, beta_temp_0);
        Cudd_RecursiveDeref(manager, beta_temp_1);
    }

    Cudd_RecursiveDeref(manager, _P);

    return beta;
}

};
