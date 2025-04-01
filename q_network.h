#pragma once
#include "network.h"
#include "deque"
#include "game.h"

struct information {
    Matrix state;
    Matrix q_values;
    double q_value;
    double reward;
    int done;
    double q_target_value;
    Matrix q_target;
    information(Matrix q_values, double q_value, double reward, int done, double q_target_value, Matrix q_target, Matrix state)
        : q_values(q_values), q_value(q_value), reward(reward), done(done), 
          q_target_value(q_target_value), q_target(q_target), state(state) {}
};

class q_network : public network {
    private:
    double gamma = 0.9;
    double epsilon = 0.5;
    int action_space_size = 4;
    public:
    q_network(int input_layer_size, std::vector <int> hidden_layers_sizes, int output_layer_size, std::vector <std::string> activationFuncions) : network(input_layer_size, hidden_layers_sizes, output_layer_size, activationFuncions) {};
    information get_information(Matrix& input, Game& game_play);
    double reward(Matrix current_state);
    void update_net(double learning_rate, int mini_batch_size, std::deque<information> experiences);
    void train(int games, int batch_size, int mini_batch_size, double learning_rate);
    int select_action(Matrix& state);

};