# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.6.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.6.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow

# Include any dependencies generated for this target.
include CMakeFiles/flow.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/flow.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/flow.dir/flags.make

CMakeFiles/flow.dir/SGMFlow.o: CMakeFiles/flow.dir/flags.make
CMakeFiles/flow.dir/SGMFlow.o: SGMFlow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/flow.dir/SGMFlow.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/flow.dir/SGMFlow.o -c /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow/SGMFlow.cpp

CMakeFiles/flow.dir/SGMFlow.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/flow.dir/SGMFlow.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow/SGMFlow.cpp > CMakeFiles/flow.dir/SGMFlow.i

CMakeFiles/flow.dir/SGMFlow.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/flow.dir/SGMFlow.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow/SGMFlow.cpp -o CMakeFiles/flow.dir/SGMFlow.s

CMakeFiles/flow.dir/SGMFlow.o.requires:

.PHONY : CMakeFiles/flow.dir/SGMFlow.o.requires

CMakeFiles/flow.dir/SGMFlow.o.provides: CMakeFiles/flow.dir/SGMFlow.o.requires
	$(MAKE) -f CMakeFiles/flow.dir/build.make CMakeFiles/flow.dir/SGMFlow.o.provides.build
.PHONY : CMakeFiles/flow.dir/SGMFlow.o.provides

CMakeFiles/flow.dir/SGMFlow.o.provides.build: CMakeFiles/flow.dir/SGMFlow.o


CMakeFiles/flow.dir/SPSFlow.o: CMakeFiles/flow.dir/flags.make
CMakeFiles/flow.dir/SPSFlow.o: SPSFlow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/flow.dir/SPSFlow.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/flow.dir/SPSFlow.o -c /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow/SPSFlow.cpp

CMakeFiles/flow.dir/SPSFlow.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/flow.dir/SPSFlow.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow/SPSFlow.cpp > CMakeFiles/flow.dir/SPSFlow.i

CMakeFiles/flow.dir/SPSFlow.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/flow.dir/SPSFlow.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow/SPSFlow.cpp -o CMakeFiles/flow.dir/SPSFlow.s

CMakeFiles/flow.dir/SPSFlow.o.requires:

.PHONY : CMakeFiles/flow.dir/SPSFlow.o.requires

CMakeFiles/flow.dir/SPSFlow.o.provides: CMakeFiles/flow.dir/SPSFlow.o.requires
	$(MAKE) -f CMakeFiles/flow.dir/build.make CMakeFiles/flow.dir/SPSFlow.o.provides.build
.PHONY : CMakeFiles/flow.dir/SPSFlow.o.provides

CMakeFiles/flow.dir/SPSFlow.o.provides.build: CMakeFiles/flow.dir/SPSFlow.o


CMakeFiles/flow.dir/flow.o: CMakeFiles/flow.dir/flags.make
CMakeFiles/flow.dir/flow.o: flow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/flow.dir/flow.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/flow.dir/flow.o -c /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow/flow.cpp

CMakeFiles/flow.dir/flow.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/flow.dir/flow.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow/flow.cpp > CMakeFiles/flow.dir/flow.i

CMakeFiles/flow.dir/flow.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/flow.dir/flow.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow/flow.cpp -o CMakeFiles/flow.dir/flow.s

CMakeFiles/flow.dir/flow.o.requires:

.PHONY : CMakeFiles/flow.dir/flow.o.requires

CMakeFiles/flow.dir/flow.o.provides: CMakeFiles/flow.dir/flow.o.requires
	$(MAKE) -f CMakeFiles/flow.dir/build.make CMakeFiles/flow.dir/flow.o.provides.build
.PHONY : CMakeFiles/flow.dir/flow.o.provides

CMakeFiles/flow.dir/flow.o.provides.build: CMakeFiles/flow.dir/flow.o


# Object files for target flow
flow_OBJECTS = \
"CMakeFiles/flow.dir/SGMFlow.o" \
"CMakeFiles/flow.dir/SPSFlow.o" \
"CMakeFiles/flow.dir/flow.o"

# External object files for target flow
flow_EXTERNAL_OBJECTS =

flow: CMakeFiles/flow.dir/SGMFlow.o
flow: CMakeFiles/flow.dir/SPSFlow.o
flow: CMakeFiles/flow.dir/flow.o
flow: CMakeFiles/flow.dir/build.make
flow: /usr/local/lib/libopencv_videostab.2.4.13.dylib
flow: /usr/local/lib/libopencv_ts.a
flow: /usr/local/lib/libopencv_superres.2.4.13.dylib
flow: /usr/local/lib/libopencv_stitching.2.4.13.dylib
flow: /usr/local/lib/libopencv_contrib.2.4.13.dylib
flow: /usr/local/lib/libopencv_nonfree.2.4.13.dylib
flow: /usr/local/lib/libopencv_ocl.2.4.13.dylib
flow: /usr/local/lib/libopencv_gpu.2.4.13.dylib
flow: /usr/local/lib/libopencv_photo.2.4.13.dylib
flow: /usr/local/lib/libopencv_objdetect.2.4.13.dylib
flow: /usr/local/lib/libopencv_legacy.2.4.13.dylib
flow: /usr/local/lib/libopencv_video.2.4.13.dylib
flow: /usr/local/lib/libopencv_ml.2.4.13.dylib
flow: /usr/local/lib/libopencv_calib3d.2.4.13.dylib
flow: /usr/local/lib/libopencv_features2d.2.4.13.dylib
flow: /usr/local/lib/libopencv_highgui.2.4.13.dylib
flow: /usr/local/lib/libopencv_imgproc.2.4.13.dylib
flow: /usr/local/lib/libopencv_flann.2.4.13.dylib
flow: /usr/local/lib/libopencv_core.2.4.13.dylib
flow: CMakeFiles/flow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable flow"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/flow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/flow.dir/build: flow

.PHONY : CMakeFiles/flow.dir/build

CMakeFiles/flow.dir/requires: CMakeFiles/flow.dir/SGMFlow.o.requires
CMakeFiles/flow.dir/requires: CMakeFiles/flow.dir/SPSFlow.o.requires
CMakeFiles/flow.dir/requires: CMakeFiles/flow.dir/flow.o.requires

.PHONY : CMakeFiles/flow.dir/requires

CMakeFiles/flow.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/flow.dir/cmake_clean.cmake
.PHONY : CMakeFiles/flow.dir/clean

CMakeFiles/flow.dir/depend:
	cd /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow /Users/Zheng/Documents/R.Urtasun/Flow/Scene-Flow/CMakeFiles/flow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/flow.dir/depend

