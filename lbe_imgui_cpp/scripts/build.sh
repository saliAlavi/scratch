mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=/home/ali/libraries/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_CXX_STANDARD=17
cmake --build . 