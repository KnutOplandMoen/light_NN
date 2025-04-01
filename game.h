#include "matrix.h"

class game { //each new game class, for instance pong or snake should inherit this class in some way, or use the same functions..
    private:
    Matrix state;
    int game_over = 0;
    public:
    Matrix get_state() {return state;}
    void take_action(Matrix action);
    double get_reward();
    int is_over();
    void train();
    void initialize();
};