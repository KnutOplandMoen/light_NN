project('GUI Project', 'cpp',
  version : '0.1',
  default_options : ['warning_level=2',
                     'cpp_std=c++20',
                     'c_std=c17',
                     'default_library=static'])

# Windows compilation has no pre-packaged SDL2 available, and compilation takes a while.
# If you want to compile from source, remove everything but the contents of the else clause.
if host_machine.system() == 'windows'
  sdl2_dep = subproject('sdl2_windows').get_variable('sdl2_windows_dep')
else
  sdl2_dep = dependency('sdl2')
endif

animationwindow_dep = dependency('animationwindow', fallback: ['animationwindow', 'animationwindow_dep'])
std_lib_facilities_dep = dependency('std_lib_facilities', fallback: ['std_lib_facilities', 'std_lib_facilities_dep'])

if host_machine.system() == 'linux'
  compiler_flags = ['-Wconversion', '-fdiagnostics-color=always', '-Werror=return-type']
else
  compiler_flags = ['-Wconversion', '-fdiagnostics-color=always', '-Werror=return-type', '-fcolor-diagnostics', '-fansi-escape-codes']
endif

src = []

exe = executable(
  'program',
  src,
  'snake_game_files/board.cpp',
  'snake_game_files/board.h',
  'snake_game_files/snake.cpp',
  'snake_game_files/snake.h',
  'functions/activation_functions.cpp',
  'functions/activation_functions.h',
  'functions/animation_functions.cpp',
  'functions/animation_functions.h',
  'functions/data_functions.cpp',
  'functions/data_functions.h',
  'functions/math_functions.cpp',
  'functions/math_functions.h',
  'functions/functions.h',
  'Utils/game.cpp',
  'Utils/game.h',
  'Utils/utils.cpp',
  'Utils/utils.h',
  'Utils/matrix.cpp',
  'Utils/matrix.h',
  'networks/network.cpp',
  'networks/network.h',
  'networks/q_network.cpp',
  'networks/q_network.h',
  'main.cpp',
  include_directories : include_directories('snake_game_files', 'functions', 'Utils', 'networks'),
  dependencies : [animationwindow_dep, sdl2_dep, std_lib_facilities_dep],
  cpp_args : compiler_flags
)
