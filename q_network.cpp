#include "q_network.h"
#include "game.h"
#include "map"
#include "math_functions.h"
#include <deque>
#include <unistd.h> 

int player_move(Game& game_play){
    while (true) {
    if (game_play.is_key_down(KeyboardKey::RIGHT)){
        std::cout << "right pressed" << std::endl;
        return 1;
    }
    else if (game_play.is_key_down(KeyboardKey::UP)){
        std::cout << "up pressed" << std::endl;
        return 0;
    }
    else if (game_play.is_key_down(KeyboardKey::DOWN)){
        std::cout << "down pressed" << std::endl;
        return 2;
    }
    else if (game_play.is_key_down(KeyboardKey::LEFT)){
        std::cout << "left pressed" << std::endl;
        return 3;
    }
}
}

int q_network::select_action(Matrix& state, Game& game_play) {
    double number = randDouble(0, 10000);

    bool from_user = false;
    
    if (!from_user) {
    if (number / 10000 < epsilon) {  
        return static_cast<int>(number) % action_space_size;  // Random action (explore)
    } else {
        Matrix q_values = feed_forward_pass(state)[0].back();
        return q_values.getMaxRow();  // Best action (exploit)
    }
    }
    else {
        int dir = player_move(game_play);
        return dir;
    }

    
}

information q_network::get_information(Matrix& state, Game& game_play) {
    bool grow = false;
    bool collision = false;
    TDT4102::Point lastPos = game_play.snake.getSnakeHead();
    Matrix q_values = feed_forward_pass(state)[0].back();

    int action = select_action(state, game_play);
    double q_value = q_values[action][0];
    Matrix prev_state = game_play.getState();

    game_play.take_action(action); //TODO: Here next state needs to be made.. in environment
    if (game_play.snake.collisionFood(game_play.foodVec) != -1){
        grow = true;
        game_play.foodVec.clear();
        game_play.newFood();
    }

    game_play.snake.move(grow);
    
    if (game_play.snake.collisionFood(game_play.foodVec) != -1){
        grow = true;
    }

    
    collision = game_play.snake.collision() || (game_play.snake.getSnakeHead() == lastPos);
    //std::cout << "collision: " << collision << std::endl;
    
    double reward = game_play.getReward(grow, collision, lastPos);
    total_reward += reward;

    int done = game_play.is_over();

    Matrix new_state = game_play.getState();

    Matrix next_action = feed_forward_pass(new_state)[0].back();
    double max_next_q_value = next_action[next_action.getMaxRow()][0];

    double q_target_value = reward + (1 - done) * gamma * max_next_q_value;

    Matrix q_target = q_values;
    q_target[action][0] = q_target_value;
    information info(q_values, q_value, reward, done, q_target_value, q_target, state);

    return info;

}

/* 
The net should be updates for when enough minibatches is done
*/


void q_network::update_net(double learning_rate, int mini_batch_size, std::deque<information> mini_batch) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<Matrix>> batch_errors;
    std::vector<std::vector<Matrix>> batch_activated_layers;

    for (int k = 0; k < mini_batch.size(); ++k) {
        information& info = mini_batch[k];
        std::vector<std::vector<Matrix>> ff = feed_forward_pass(info.state);
        std::vector<Matrix> activated_layers = ff[0];
        activated_layers.insert(activated_layers.begin(), info.state); 
        std::vector<Matrix> error = get_errors(info.state, info.q_target); 
        batch_errors.push_back(error);
        batch_activated_layers.push_back(activated_layers);
    }

    gradient_descent_weights(batch_errors, learning_rate, mini_batch[0].state, batch_activated_layers);
    gradient_descent_biases(batch_errors, learning_rate, mini_batch[0].state, batch_activated_layers);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
}

void q_network::train(int games, int batch_size, int mini_batch_size, double learning_rate) {
    std::cout << "starting training" << std::endl;
    std::deque<information> experiences;
    for (int game = 0; game < games; ++game) {
        std::cout << "Game: " << game+1 << std::endl;
        Game game_play;
        int move = 0;
        total_reward = 0;
        while (!game_play.is_over()) {

            game_play.drawBoard();
            Matrix state = game_play.getState();
            information info = get_information(state, game_play);

            if (experiences.size() >= batch_size) {
                experiences.pop_front();
            }

            if (experiences.size() >= mini_batch_size) {
                std::vector<int> indices(experiences.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
                
                std::deque<information> mini_batch;
                for (int i = 0; i < mini_batch_size && i < experiences.size(); ++i) {
                    mini_batch.push_back(experiences[indices[i]]);
                }
                update_net(learning_rate, mini_batch_size, mini_batch);  // Train on a single mini-batch
            }
            
            epsilon = std::max(0.1, epsilon * 0.9999); // Decay epsilon over time, with a minimum value of 0.1
            experiences.push_back(info);
            game_play.next_frame();
        }
        std::cout << "game: " << game << "/ " << games << " finished" << std::endl; 
        std::cout << "snake size: " << game_play.snake.getSnakeBody().size() << std::endl;
        std::cout << "total reward: " << total_reward << std::endl;
        std::cout << "epsilon: " << epsilon << std::endl;
        game_play.close();

    }
}