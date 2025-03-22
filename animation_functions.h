#include <vector>
#include "network.h"
#include "AnimationWindow.h"

void update(std::vector <double>& epochs_n, std::vector <double>& loss_n, std::vector <double>& accuracy_n, double current_accuracy, double current_loss, int width, int height, int epochs, TDT4102::AnimationWindow& window);