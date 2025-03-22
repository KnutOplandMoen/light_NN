#include <vector>
#include "network.h"
#include "AnimationWindow.h"

void update(std::vector <double>& epochs_n, std::vector <double>& loss_n, std::vector <double>& accuracy_n, double current_accuracy, double current_loss, int width, int height, int epochs, TDT4102::AnimationWindow& window);
void visualize_feed_forward(std::vector<Matrix> activated_layers, Matrix x_labels);
void visualize_feed_forward_multiple(std::vector<std::vector<Matrix>> multiple_activated_layers, std::vector<Matrix> x_labels);