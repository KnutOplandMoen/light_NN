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
    double gamma = 0.99;
    double epsilon = 1.0;
    double epsilon_decay = 0.99;
    double min_epsilon = 0.01;
    int action_space_size = 4;
    double total_reward = 0;
    public:
    q_network(int input_layer_size, std::vector <int> hidden_layers_sizes, int output_layer_size, std::vector <std::string> activationFuncions) : network(input_layer_size, hidden_layers_sizes, output_layer_size, activationFuncions) {};
    information get_information(Matrix& input, Game& game_play);
    double reward(Matrix current_state);
    void update_net(double learning_rate, int mini_batch_size, std::deque<information> experiences);
    void train(int games, int batch_size, int mini_batch_size, double learning_rate, std::map<std::string, int> autosave_file = {});
    int select_action(Matrix& state, Game& game_play);
    double get_epsilon() {return epsilon;}

    /**
     * @brief Set desiered epsilon (0->1)
     * 
     * @param epsilon_set Epsilon to be set.
     */
    void set_epsilon(double epsilon_set) {epsilon = epsilon_set;}

    /**
     * @brief Set desiered minimum epsilon (0->1)
     * 
     * @param epsilon_set Minimum epsilon to be set.
     */
    void set_epsilon_min(double epsilon_min_set) {min_epsilon = epsilon_min_set;}


    void play(int games);

};