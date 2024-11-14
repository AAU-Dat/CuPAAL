#include <stdio.h> 

void fit_Parameters(char* filepath, float** training_set, char** parameters) {
    printf("Fitting parameters\n");
    printf("Filepath: %s\n", filepath);
    printf("Training set: %f\n", training_set[0][0]);
    printf("Parameters: %s\n", parameters[0]);
}

int main()
{
    return 1;
}