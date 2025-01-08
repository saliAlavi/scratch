sudo apt update
sudo apt install cmake xorg-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev libglfw3-dev
sudo apt-get install libglfw3-dev libglew-dev
vcpkg integrate install
vcpkg install imgui[glfw-binding]
vcpkg install glfw3 glew imgui
vcpkg install imgui[opengl3-binding]