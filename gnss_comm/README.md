# gnss_comm

**Authors/Maintainers:** CAO Shaozu (shaozu.cao AT gmail.com)

The *gnss_comm* package contains basic definitions and utility functions for GNSS raw measurement processing. 

## 1. Prerequisites

### 1.1 C++11 Compiler
This package requires some features of C++11.

### 1.2 ROS
This package is developed under [ROS2 Humble](https://docs.ros.org/en/humble/index.html) environment.

### 1.3 Eigen
Our code uses [Eigen 3.3.3](https://gitlab.com/libeigen/eigen/-/archive/3.3.3/eigen-3.3.3.zip) for matrix manipulation. After downloading and unzipping the Eigen source code package, you may install it with the following commands:

```
cd eigen-3.3.3/
mkdir build
cd build
cmake ..
sudo make install
```

### 1.4 Glog
We use google's glog library for message output. If you are using Ubuntu, install it by:
```
sudo apt-get install libgoogle-glog-dev
```
If you are on other OS or just want to build it from source, please follow [these instructions](https://github.com/google/glog#building-glog-with-cmake) to install it.


## 2. Build gnss_comm library
Clone the repository to your ros2 workspace (for example `~/ros2_ws/`):
```
cd ~/ros2_ws/src/
git clone
```
Then build the package with:
```
cd ~/ros2_ws/
colcon build
```
If you encounter any problem during the building of *gnss_comm*, try with docker in [the next section](#docker_section).

## 3. <a name="docker_section"></a>Docker Support
TODO

## 4. Acknowledgements
Many of the definitions and utility functions in this package are adapted from [RTKLIB](http://www.rtklib.com/).

## 5. License
The source code is released under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html) license.
