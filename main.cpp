#include <iostream>
#include "MatLib.h"
using namespace std;

int main(){
    string loss_type = "MAE";
    int epoch = 200; float lr = 0.05f;
    vector<int> hidden_nodes = {128};
    Mat X, Y, X_Val, Y_Val;
    X.loadmat("Dataset2/X_Train_House.txt");
    Y.loadmat("Dataset2/Y_Train_House.txt");
    X_Val.loadmat("Dataset2/X_Valid_House.txt");
    Y_Val.loadmat("Dataset2/Y_Valid_House.txt");
    vector<Mat> W(hidden_nodes.size() + 1), B(hidden_nodes.size() + 1);
    Loss_History history;
    Mat Y_mean(1,Y.col), Y_std(1,Y.col);
    FeatureScaling(Y, Y, Y_mean, Y_std);
    FeatureScaling(Y_Val, Y_mean, Y_std);
    StartTimer();
    MLP(X, Y, X_Val,Y_Val, W, B, hidden_nodes, lr, epoch, history, loss_type , 0.5f, 30);
    StopTimer(); history.print_final(); PrintTimer();

    // Load test set
    Mat X_Test, Y_Test;
    X_Test.loadmat("Dataset2/X_Test_House.txt");
    Y_Test.loadmat("Dataset2/Y_Test_House.txt");

    Mat Y_Pred(Y_Test.row, Y_Test.col);
    Forward_Pass_ReLU(X_Test, W, B, Y_Pred, loss_type);
    Rescale_Y(Y_Pred, Y_mean, Y_std);
	// ShowSoftmaxPredict(Y_Pred, Y_Test);
    ModelEvaluation(Y_Test, Y_Pred, loss_type);
    Y_Pred.savetxt("Dataset2/Y_Pred_House.txt");
}