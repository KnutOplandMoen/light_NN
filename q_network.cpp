#include "q_network.h"
#include "game.h"
#include "map"

information q_network::get_information(Matrix input, int done, game game_play) {

    Matrix q_values = feed_forward_pass(input)[0].back();
    double q_value = q_values[q_values.getMaxRow()][0];

    game_play.take_action(q_values); //TODO: Here next state needs to be made.. in environment
    Matrix prev_state = game_play.get_state();
    double reward = game_play.get_reward();
    int done = game_play.is_over();

    Matrix next_action = feed_forward_pass(prev_state)[0].back();
    double max_next_q_value = next_action[next_action.getMaxRow()][0];

    double q_target_value = reward + (1 - done) * gamma * max_next_q_value;

    Matrix q_target = q_values;
    q_target[q_values.getMaxRow()][0] = q_target_value;
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

void q_network::train() {

    /*
    Plan: 

    Initialize loop with number of games
        For each game initialize a game
        do actions
        save experiences
        when enough actions is take (experience_size = X): 
            train network on experiences using update_network
        
    */
}