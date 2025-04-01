#include "network.h"

class q_network : public network {
    private:
    double gamma;
    public:
    q_network(int input_layer_size, std::vector <int> hidden_layers_sizes, int output_layer_size, std::vector <std::string> activationFuncions) : network(input_layer_size, hidden_layers_sizes, output_layer_size, activationFuncions) {};
    double q_network::get_loss(Matrix input, int done, game game_play);
    double q_network::reward(Matrix current_state);
    void update_net(int epochs, double learning_rate, int batch_size, std::vector<Matrix> experiences);
};