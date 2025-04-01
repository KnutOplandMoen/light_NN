#include "q_network.h"
#include "game.h"
#include "map"
#include "math_functions.h"
#include <deque>

int q_network::select_action(Matrix& state) {
    double number = randDouble(0, 10000);
    if (number / 10000 < epsilon) {  
        return rand() % action_space_size;  // Random action (explore)
    } else {
        Matrix q_values = feed_forward_pass(state)[0].back();
        return q_values.getMaxRow();  // Best action (exploit)
    }
}

information q_network::get_information(Matrix& state, int done) {

    Matrix q_values = feed_forward_pass(state)[0].back();

    int action = select_action(state);

    double q_value = q_values[action][0];
    Matrix prev_state = game_play.get_state();

    game_play.take_action(action); //TODO: Here next state needs to be made.. in environment
    double reward = game_play.get_reward();
    int done = game_play.is_over();

    Matrix new_state = game_play.get_state();

    Matrix next_action = feed_forward_pass(new_state)[0].back();
    double max_next_q_value = next_action[next_action.getMaxRow()][0];

    double q_target_value = reward + (1 - done) * gamma * max_next_q_value;

    Matrix q_target = q_values;
    q_target[action][0] = q_target_value;
    information info(q_values, q_value, reward, done, q_target_value, q_target);

    return info;

}

/* 
The net should be updates for when enough minibatches is done
*/


void q_network::update_net(int epochs, double learning_rate, int batch_size, std::deque<information> experiences) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto start = std::chrono::high_resolution_clock::now();

        std::random_shuffle(experiences.begin(), experiences.end());
        
        for (int j = 0; j < experiences.size(); j += batch_size) {
            std::vector<std::vector<Matrix>> batch_errors;
            std::vector<std::vector<Matrix>> batch_activated_layers;

            for (int k = 0; k < batch_size && (j + k) < experiences.size(); ++k) {
                information& info = experiences[j + k];

                std::vector<std::vector<Matrix>> ff = feed_forward_pass(info.state);
                std::vector<Matrix> activated_layers = ff[0];
                activated_layers.insert(activated_layers.begin(), info.state); 
                std::vector<Matrix> error = get_errors(info.state, info.q_target); 
                batch_errors.push_back(error);
                batch_activated_layers.push_back(activated_layers);
            }

            gradient_descent_weights(batch_errors, learning_rate, experiences[j].state, batch_activated_layers);
            gradient_descent_biases(batch_errors, learning_rate, experiences[j].state, batch_activated_layers);
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Q-learning Epoch " << epoch + 1 << " finished in " << duration.count() / 1000.0 << " seconds.\n";
    }
}


void q_network::train(int games, int batch_size, game& game_play) {
    /*
    Initialize loop with number of games
        For each game initialize a game
        do actions
        save experiences
        when enough actions is take (experience_size = X): 
            train network on experiences using update_network
        
    */

    for (int game = 0; game < games; ++game) {
        game_play.initialize();

        std::deque<information> experiences;
        int move = 0;
        while (!game_play.is_over()) {

            Matrix state = game_play.get_state();
            int done = game_play.is_over();
            information info = get_information(state, done, game_play);

            if (experiences.size() >= batch_size) {
                update_net(games, 0.8, batch_size, experiences);
                experiences.pop_front();
            } 
            experiences.push_back(info);
        }

    }
}